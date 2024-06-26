from langchain_core.prompts import FewShotPromptTemplate,PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_experimental.tabular_synthetic_data.openai import (
    OPENAI_TEMPLATE,
    create_openai_data_generator
    )
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX
)
from langchain_openai import ChatOpenAI

class Order(BaseModel):
    order_id: str
    customer_name: str
    table_number: int
    order_date: str
    order_time: str
    waiter_id: str
    item_id: str
    item_name: str
    quantity: int
    item_price: float
    total_item_price: float
    order_status: str
    payment_method: str
    total_order_price: float
    discounts: float
    tax: float
    tip: float
    payment_status: str


examples=[
    {
        "example":"""
                  Order ID: 001, Customer Name: John, Table Number: 12, Order Date: 2024-06-26, Order Time: 19:00, Waiter ID: W001, Item ID: I001, Item Name: Cheeseburger, Quantity: 2, Item Price: 12.50, Total Item Price: 25.00, Order Status: Completed, Payment Method: Credit Card, Total Order Price: 60.00, Discounts: 0.00, Tax: 5.00, Tip: 5.00, Payment Status: Paid
                  """

    },
    {
        "example":"""
              Order ID: 002, Customer Name: Prince, Table Number: 5, Order Date: 2024-06-26, Order Time: 19:15, Waiter ID: W002, Item ID: I002, Item Name: Vegan Salad, Quantity: 1, Item Price: 9.00, Total Item Price: 9.00, Order Status: Completed, Payment Method: Cash, Total Order Price: 45.00, Discounts: 0.00, Tax: 3.60, Tip: 4.00, Payment Status: Paid
             """
    },
    {
    "example":"""Order ID: 003, Customer Name: Bob, Table Number: 20, Order Date: 2024-06-26, Order Time: 20:10, Waiter ID: W003, Item ID: I003, Item Name: Grilled Chicken, Quantity: 3, Item Price: 15.00, Total Item Price: 45.00, Order Status: Pending, Payment Method: Online Payment, Total Order Price: 53.40, Discounts: 2.40, Tax: 6.00, Tip: 0.00, Payment Status: Pending"""
    },
    {
        "example":"""Order ID: 004, Customer Name: John: David, Table Number: 8, Order Date: 2024-06-26, Order Time: 18:45, Waiter ID: W001, Item ID: I001, Item Name: Cheeseburger, Quantity: 1, Item Price: 12.50, Total Item Price: 12.50, Order Status: Completed, Payment Method: Credit Card, Total Order Price: 28.75, Discounts: 0.00, Tax: 1.25, Tip: 2.50, Payment Status: Paid"""
    },
    {
        "example":"""Order ID: 005, Customer Name: John: Marrie, Table Number: 15, Order Date: 2024-06-26, Order Time: 21:00, Waiter ID: W002, Item ID: I004, Item Name: Fish Tacos, Quantity: 2, Item Price: 10.00, Total Item Price: 20.00, Order Status: Completed, Payment Method: Cash, Total Order Price: 24.40, Discounts: 0.00, Tax: 1.20, Tip: 3.20, Payment Status: Paid"""
    }

]
OPENAI_TEMPLATE=PromptTemplate(input_variables=["example"],template="{example}")

prompt_template=FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["subject","extra"],
    example_prompt=OPENAI_TEMPLATE
)
syntghetic_data_generator=create_openai_data_generator(
    output_schema=Order,
    llm=ChatOpenAI(model="gpt-4o"),
    prompt=prompt_template
)

results=syntghetic_data_generator.generate(subject="Order History",
                                           extra="customer name must Indian names ,generate Indian names only",
                                           runs=20
                                           )
for data in results:
    print(data.customer_name)