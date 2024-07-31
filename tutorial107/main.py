from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from  datetime import datetime

rate_limiters=InMemoryRateLimiter(
    requests_per_second = .05,
    check_every_n_seconds = 0.1,
    max_bucket_size  = 10
)
llm=ChatOpenAI(model="gpt-4o",rate_limiter=rate_limiters)
for i in range(10):
    response=llm.invoke("hello")
    print(response.content)
    print(datetime.now())


