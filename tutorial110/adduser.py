import boto3
client=boto3.client("iam")

def createuser(user):
    response=client.create_user(UserName=user)
    return response