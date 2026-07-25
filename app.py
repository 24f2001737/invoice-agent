import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request

from decision_engine import decide_packages
from storage import (
    canonical_json,
    create_task,
    get_message,
    get_task_for_principal,
    get_proposals,
    init_db,
    insert_proposal,
    list_tasks,
    sha256_json,
    update_task,
)


app = Flask(__name__)

init_db()

DB_OPERATION_LOCK = threading.RLock()

PROFILE = "ga5-invoice-action-agent/v1"

INPUT_MEDIA = (
    "application/vnd.ga5.invoice-claim-batch+json"
)

PROPOSAL_MEDIA = (
    "application/vnd.ga5.invoice-action-proposals+json"
)

RESULT_MEDIA = (
    "application/vnd.ga5.invoice-action-results+json"
)

RECEIPT_MEDIA = (
    "application/vnd.ga5.invoice-action-receipts+json"
)

A2A_VERSION = "1.0"


def now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def principal_from_request():
    header = request.headers.get(
        "Authorization",
        "",
    )

    if not header.startswith(
        "Bearer "
    ):
        return None

    token = header[
        len("Bearer "):
    ].strip()

    if not token:
        return None

    return hashlib.sha256(
        token.encode("utf-8")
    ).hexdigest()


def error_response(
    message,
    status=400,
    code=None,
):
    body = {
        "error": message
    }

    if code:
        body["code"] = code

    return jsonify(body), status


def require_a2a_headers():
    version = request.headers.get(
        "A2A-Version"
    )

    if version != A2A_VERSION:
        return error_response(
            "Unsupported A2A version.",
            400,
        )

    content_type = request.headers.get(
        "Content-Type",
        "",
    )

    if not content_type.startswith(
        "application/a2a+json"
    ):
        return error_response(
            "Unsupported media type.",
            415,
        )

    principal = principal_from_request()

    if principal is None:
        return error_response(
            "Authentication required.",
            401,
        )

    return principal


def task_response(task):
    response = jsonify(
        {
            "task": task
        }
    )

    response.headers[
        "Content-Type"
    ] = "application/a2a+json"

    return response


def build_agent_card():
    base_url = (
        os.environ.get(
            "PUBLIC_BASE_URL",
            "https://example.com/a2a",
        )
        .rstrip("/")
    )

    return {
        "name": "Invoice Action Agent",
        "description": (
            "AI-powered invoice decision agent "
            "that proposes safe invoice actions "
            "and executes only accepted proposals."
        ),
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
        },
        "supportedInterfaces": [
            {
                "url": base_url,
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": "1.0",
            }
        ],
        "defaultInputModes": [
            INPUT_MEDIA
        ],
        "defaultOutputModes": [
            PROPOSAL_MEDIA,
            RECEIPT_MEDIA,
        ],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "invoice_action_agent",
                "description": (
                    "Reads invoice packages, "
                    "proposes one safe action per invoice, "
                    "cites evidence, and processes "
                    "grader-approved results."
                ),
                "tags": [
                    "invoice",
                    "finance",
                    "reconciliation",
                    "approval",
                ],
            }
        ],
    }


@app.get(
    "/.well-known/agent-card.json"
)
def agent_card():
    response = jsonify(
        build_agent_card()
    )

    response.headers[
        "Content-Type"
    ] = "application/a2a+json"

    return response


def make_initial_task(
    principal,
    message,
    batch,
    proposals,
):
    message_id = message[
        "messageId"
    ]

    context_id = (
        message.get(
            "contextId"
        )
        or "ctx-" + uuid.uuid4().hex
    )

    message_hash = sha256_json(
        message
    )

    task_id = (
        "task-" + uuid.uuid4().hex
    )

    history = [
        message
    ]

    artifact = {
        "name": "invoice-action-proposals",
        "parts": [
            {
                "mediaType": PROPOSAL_MEDIA,
                "data": {
                    "batchId": batch[
                        "batchId"
                    ],
                    "proposals": proposals,
                },
            }
        ],
    }

    task = {
        "id": task_id,
        "contextId": context_id,
        "status": {
            "state": "TASK_STATE_INPUT_REQUIRED"
        },
        "history": history,
        "artifacts": [
            artifact
        ],
    }

    return (
        task_id,
        context_id,
        message_hash,
        task,
    )


def validate_initial_message(message):
    if not isinstance(
        message,
        dict,
    ):
        return False

    if message.get(
        "role"
    ) != "ROLE_USER":
        return False

    if not isinstance(
        message.get(
            "messageId"
        ),
        str,
    ):
        return False

    parts = message.get(
        "parts"
    )

    if (
        not isinstance(parts, list)
        or len(parts) != 1
    ):
        return False

    part = parts[0]

    if part.get(
        "mediaType"
    ) != INPUT_MEDIA:
        return False

    if not isinstance(
        part.get("data"),
        dict,
    ):
        return False

    return True


@app.post(
    "/a2a/message:send"
)
def send_message():
    principal = require_a2a_headers()

    if not isinstance(
        principal,
        str,
    ):
        return principal

    body = request.get_json(
        silent=True
    )

    if not isinstance(
        body,
        dict,
    ):
        return error_response(
            "Malformed request.",
            400,
        )

    message = body.get(
        "message"
    )

    if not validate_initial_message(
        message
    ):
        # Try continuation before rejecting.
        if (
            isinstance(
                message,
                dict,
            )
            and message.get(
                "taskId"
            )
        ):
            return handle_continuation(
                principal,
                body,
            )

        return error_response(
            "Malformed request.",
            400,
        )

    message_id = message[
        "messageId"
    ]

    message_hash = sha256_json(
        message
    )

    # Idempotency check.
    existing = get_message(
        principal,
        message_id,
    )

    if existing:
        if existing[
            "message_hash"
        ] != message_hash:
            return error_response(
                "Request conflicts with an existing message.",
                409,
                "IDEMPOTENCY_CONFLICT",
            )

        task = get_task_for_principal(
            existing["task_id"],
            principal,
        )

        if not task:
            return error_response(
                "Request could not be completed.",
                500,
            )

        return task_response(
            task["task"]
        )

    data = message[
        "parts"
    ][0]["data"]

    batch_id = data.get(
        "batchId"
    )

    packages = data.get(
        "packages"
    )

    if (
        not isinstance(
            batch_id,
            str,
        )
        or not isinstance(
            packages,
            list,
        )
        or not packages
    ):
        return error_response(
            "Malformed request.",
            400,
        )

    package_ids = [
        package.get(
            "packageId"
        )
        for package in packages
    ]

    if (
        any(
            not isinstance(
                x,
                str,
            )
            for x in package_ids
        )
        or len(
            package_ids
        )
        != len(
            set(package_ids)
        )
    ):
        return error_response(
            "Malformed request.",
            400,
        )

    with DB_OPERATION_LOCK:
        # Recheck after acquiring lock to prevent
        # concurrent duplicate model work.
        existing = get_message(
            principal,
            message_id,
        )

        if existing:
            if existing[
                "message_hash"
            ] != message_hash:
                return error_response(
                    "Request conflicts with an existing message.",
                    409,
                    "IDEMPOTENCY_CONFLICT",
                )

            task = get_task_for_principal(
                existing[
                    "task_id"
                ],
                principal,
            )

            return task_response(
                task["task"]
            )

        decisions = decide_packages(
            packages
        )

        proposals = []

        for decision in decisions:
            action_id = (
                "action-"
                + uuid.uuid4().hex
            )

            proposal = {
                "packageId": decision[
                    "packageId"
                ],
                "actionId": action_id,
                "action": decision[
                    "action"
                ],
                "facts": {
                    "vendorName": decision.get(
                        "vendorName",
                        "",
                    ),
                    "invoiceNumber": decision.get(
                        "invoiceNumber",
                        "",
                    ),
                    "amountMinor": decision.get(
                        "amountMinor",
                        0,
                    ),
                    "currency": decision.get(
                        "currency",
                        "INR",
                    ),
                },
                "evidenceRefs": decision[
                    "evidenceRefs"
                ],
                "rationale": decision[
                    "rationale"
                ],
            }

            proposals.append(
                proposal
            )

        (
            task_id,
            context_id,
            _,
            task,
        ) = make_initial_task(
            principal,
            message,
            data,
            proposals,
        )

        task["id"] = task_id

        create_task(
            principal=principal,
            message_id=message_id,
            message_hash=message_hash,
            batch_id=batch_id,
            context_id=context_id,
            task_json=task,
            created_at=now(),
        )

        for proposal in proposals:
            insert_proposal(
                task_id,
                proposal[
                    "packageId"
                ],
                proposal[
                    "actionId"
                ],
                proposal,
            )

    return task_response(
        task
    )


def handle_continuation(
    principal,
    body,
):
    message = body.get(
        "message"
    )

    if not isinstance(
        message,
        dict,
    ):
        return error_response(
            "Malformed request.",
            400,
        )

    task_id = message.get(
        "taskId"
    )

    context_id = message.get(
        "contextId"
    )

    parts = message.get(
        "parts"
    )

    if (
        not isinstance(
            task_id,
            str,
        )
        or not isinstance(
            context_id,
            str,
        )
        or not isinstance(
            parts,
            list,
        )
        or len(parts) != 1
    ):
        return error_response(
            "Malformed request.",
            400,
        )

    part = parts[0]

    if part.get(
        "mediaType"
    ) != RESULT_MEDIA:
        return error_response(
            "Malformed request.",
            400,
        )

    data = part.get(
        "data"
    )

    if not isinstance(
        data,
        dict,
    ):
        return error_response(
            "Malformed request.",
            400,
        )

    with DB_OPERATION_LOCK:
        stored = get_task_for_principal(
            task_id,
            principal,
        )

        if not stored:
            return error_response(
                "Task not found.",
                404,
            )

        task = stored[
            "task"
        ]

        if (
            task.get(
                "contextId"
            )
            != context_id
        ):
            return error_response(
                "Invalid task context.",
                409,
            )

        if task[
            "status"
        ][
            "state"
        ] in (
            "TASK_STATE_COMPLETED",
            "TASK_STATE_CANCELED",
        ):
            return task_response(
                task
            )

        original_message = task[
            "history"
        ][0]

        original_data = (
            original_message[
                "parts"
            ][0]["data"]
        )

        if data.get(
            "batchId"
        ) != original_data.get(
            "batchId"
        ):
            return error_response(
                "Invalid task continuation.",
                409,
            )

        stored_proposals = {
            item[
                "package_id"
            ]: item[
                "proposal"
            ]
            for item in get_proposals(
                task_id
            )
        }

        results = data.get(
            "results"
        )

        if not isinstance(
            results,
            list,
        ):
            return error_response(
                "Malformed request.",
                400,
            )

        if len(
            results
        ) != len(
            stored_proposals
        ):
            return error_response(
                "Invalid task continuation.",
                409,
            )

        seen = set()

        for result in results:
            package_id = result.get(
                "packageId"
            )

            if (
                package_id in seen
                or package_id
                not in stored_proposals
            ):
                return error_response(
                    "Invalid task continuation.",
                    409,
                )

            seen.add(
                package_id
            )

            proposal = stored_proposals[
                package_id
            ]

            if (
                result.get(
                    "actionId"
                )
                != proposal[
                    "actionId"
                ]
                or result.get(
                    "action"
                )
                != proposal[
                    "action"
                ]
            ):
                return error_response(
                    "Invalid task continuation.",
                    409,
                )

            if result.get(
                "outcome"
            ) not in (
                "ACCEPTED",
                "REJECTED",
            ):
                return error_response(
                    "Malformed request.",
                    400,
                )

            if not isinstance(
                result.get(
                    "receiptNonce"
                ),
                str,
            ):
                return error_response(
                    "Malformed request.",
                    400,
                )

        executions = []

        for result in results:
            if result[
                "outcome"
            ] != "ACCEPTED":
                continue

            proposal = stored_proposals[
                result[
                    "packageId"
                ]
            ]

            executions.append(
                {
                    "packageId": proposal[
                        "packageId"
                    ],
                    "actionId": proposal[
                        "actionId"
                    ],
                    "action": proposal[
                        "action"
                    ],
                    "receiptNonce": result[
                        "receiptNonce"
                    ],
                    "facts": proposal[
                        "facts"
                    ],
                    "evidenceRefs": proposal[
                        "evidenceRefs"
                    ],
                }
            )

        task["history"].append(
            message
        )

        task["artifacts"].append(
            {
                "name": "invoice-action-receipts",
                "parts": [
                    {
                        "mediaType": RECEIPT_MEDIA,
                        "data": {
                            "batchId": data[
                                "batchId"
                            ],
                            "executions": executions,
                        },
                    }
                ],
            }
        )

        task["status"] = {
            "state": "TASK_STATE_COMPLETED"
        }

        update_task(
            task_id,
            "TASK_STATE_COMPLETED",
            task,
            now(),
        )

    return task_response(
        task
    )


@app.get(
    "/a2a/tasks/<task_id>"
)
def get_task_route(
    task_id
):
    principal = require_a2a_headers()

    if not isinstance(
        principal,
        str,
    ):
        return principal

    task = get_task_for_principal(
        task_id,
        principal,
    )

    if not task:
        return error_response(
            "Task not found.",
            404,
        )

    return task_response(
        task["task"]
    )


@app.get(
    "/a2a/tasks"
)
def list_tasks_route():
    principal = require_a2a_headers()

    if not isinstance(
        principal,
        str,
    ):
        return principal

    response = jsonify(
        {
            "tasks": list_tasks(
                principal
            )
        }
    )

    response.headers[
        "Content-Type"
    ] = "application/a2a+json"

    return response


@app.post(
    "/a2a/tasks/<task_id>:cancel"
)
def cancel_task(
    task_id
):
    principal = require_a2a_headers()

    if not isinstance(
        principal,
        str,
    ):
        return principal

    with DB_OPERATION_LOCK:
        stored = get_task_for_principal(
            task_id,
            principal,
        )

        if not stored:
            return error_response(
                "Task not found.",
                404,
            )

        task = stored[
            "task"
        ]

        state = task[
            "status"
        ][
            "state"
        ]

        if state in (
            "TASK_STATE_COMPLETED",
            "TASK_STATE_CANCELED",
        ):
            return error_response(
                "Task is already terminal.",
                409,
            )

        task["status"] = {
            "state": "TASK_STATE_CANCELED"
        }

        task["history"].append(
            {
                "role": "ROLE_USER",
                "parts": [
                    {
                        "mediaType": (
                            "application/a2a+json"
                        ),
                        "data": {
                            "type": "cancel"
                        },
                    }
                ],
            }
        )

        update_task(
            task_id,
            "TASK_STATE_CANCELED",
            task,
            now(),
        )

    return task_response(
        task
    )


@app.get("/")
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "invoice-action-agent",
        }
    )


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000,
            )
        ),
    )
