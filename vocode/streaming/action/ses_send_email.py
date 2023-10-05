from typing import Optional, Type

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field

from vocode import getenv
from vocode.streaming.action.base_action import BaseAction
from vocode.streaming.models.actions import (
    ActionConfig,
    ActionInput,
    ActionOutput,
    ActionType,
)


class AmazonSESSendEmailActionConfig(
    ActionConfig, type=ActionType.AMAZON_SES_SEND_EMAIL
):
    pass


class AmazonSESSendEmailParameters(BaseModel):
    recipient_email: str = Field(..., description="The email address of the recipient.")
    body: str = Field(..., description="The body of the email.")
    subject: Optional[str] = Field(None, description="The subject of the email.")


class AmazonSESSendEmailResponse(BaseModel):
    success: bool


class AmazonSESSendEmail(
    BaseAction[
        AmazonSESSendEmailActionConfig,
        AmazonSESSendEmailParameters,
        AmazonSESSendEmailResponse,
    ]
):
    description: str = "Sends an email using Amazon SES."
    action_type: str = ActionType.AMAZON_SES_SEND_EMAIL
    parameters_type: Type[AmazonSESSendEmailParameters] = AmazonSESSendEmailParameters
    response_type: Type[AmazonSESSendEmailResponse] = AmazonSESSendEmailResponse

    async def run(
        self, action_input: ActionInput[AmazonSESSendEmailParameters]
    ) -> ActionOutput[AmazonSESSendEmailResponse]:
        ses = boto3.client("ses", region_name=getenv("AWS_REGION"))

        # Create the email
        email_subject = (
            action_input.params.subject
            if action_input.params.subject
            else "Email from Vocode"
        )

        try:
            response = ses.send_email(
                Source=getenv("SES_SENDER_EMAIL"),
                Destination={
                    "ToAddresses": [action_input.params.recipient_email.strip()]
                },
                Message={
                    "Subject": {"Data": email_subject},
                    "Body": {"Text": {"Data": action_input.params.body}},
                },
            )
            print("Email sent! Message ID:", response["MessageId"])
            return ActionOutput(
                action_type=self.action_config.type,
                response=AmazonSESSendEmailResponse(success=True),
            )
        except ClientError as e:
            print("Error sending email:", e.response["Error"]["Message"])
            return ActionOutput(
                action_type=self.action_config.type,
                response=AmazonSESSendEmailResponse(success=False),
            )
