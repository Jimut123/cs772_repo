from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# The higher the temperature, the higher the randomness of the output,
# and vice-versa. If you want your responses to be more focused and deterministic, 
# go for the lower temperature value. And if you want it to be more creative, use a higher value. 
# The temperature value ranges between 0 and 2. 


response = client.chat.completions.create(
  model = "gpt-3.5-turbo",
  temperature = 0.8,
  max_tokens = 3000,
  response_format={ "type": "json_object" },
  messages = [
    {"role": "system", "content": "You are a funny comedian who tells dad jokes. The output should be in JSON format."},
    {"role": "user", "content": "Write a dad joke related to numbers."},
    {"role": "assistant", "content": "Q: How do you make 7 even? A: Take away the s."},
    {"role": "user", "content": "Write one related to programmers."}
  ]
)

        
print(response.choices[0].message.content)



load_dotenv()
client = OpenAI()

# This is  text completion example
response = client.chat.completions.create(
  model = "gpt-3.5-turbo",
  temperature = 0.8,
  max_tokens = 3000,
  messages = [
    {"role": "system", "content": "You are a poet who creates poems that evoke emotions."},
    {"role": "user", "content": "Write a short poem for programmers."}
  ]
)

print(response.choices[0].message.content)