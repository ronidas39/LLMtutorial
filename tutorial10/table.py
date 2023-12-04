import boto3
dbrs=boto3.resource("dynamodb")
table=dbrs.create_table(TableName="msgTable",\
                        KeySchema=[{"AttributeName":"SessionId","KeyType":"HASH"}],
                        AttributeDefinitions=[{"AttributeName":"SessionId","AttributeType":"S"}],
                        BillingMode="PAY_PER_REQUEST"
                        )