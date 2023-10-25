from langchain.adapters import openai
#dic to msg
msg=openai.convert_dict_to_message({"role":"system","content":"you are an intelligent assistant who can answer anything very smartly"})
print(msg)

#dic to msgessage_to_dict(msg)
dic=openai.convert_m
print(dic)