
from diagrams import Diagram
from diagrams.aws.compute import Lambda
from diagrams.aws.compute import EC2
from diagrams.aws.integration import SNS

with Diagram("Cloud Architecture", show=False):
    ec2 = EC2("EC2 Instance")
    lambda_function = Lambda("Lambda Function")
    sns = SNS("SNS Topic")

    ec2 >> lambda_function >> sns
