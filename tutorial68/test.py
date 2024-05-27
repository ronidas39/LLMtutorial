from langchain_experimental.data_anonymizer import PresidioAnonymizer
anonymizer=PresidioAnonymizer()
sample="my name is john smith , i am from USA , my email id is jsmith@frekiu.com"
data=anonymizer.anonymize(sample)
print(data)