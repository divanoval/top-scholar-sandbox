# Environment Description

You are currently running on an H100 SXM with 500 GB of disk space and 16 vCPU cores available. You will be running on this machine continuously so you can store anything that you want on it locally.

Code should all be written in Python and you should use uv to manage your Python environment and packages. If you install any further non-python software packages make sure to mention these in your workspace readme. 

YOU ARE RUNNING AUTONOMOUSLY AND SHOULD EXECUTE THIS PROJECT WITHOUT ANY USER INTERVENTION. DO NOT ASK FOR INPUT FROM THE USER, DO NOT ENTER PLAN MODE. If you ask for input, your question will not be answered and you will just be told to continue. You should make all project decisions on your own. 

## Github

All code should be pushed to Github, using your username `GITHUB_USER` and fine-grained token `GITHUB_TOKEN`, both of which can be found in `.env`.

## OpenRouter

You have also been given an Openrouter API key in the `.env` file with $500 of credits available. Experiments that you run will use up credits, so you should check frequently how many credits that you have remaining and make sure that you budget accordingly.

To check the remaining number of credits, make a call to the OpenRouter API using your API key and making the following request:
```
import requests
import json

response = requests.get(
  url="https://openrouter.ai/api/v1/key",
  headers={
    "Authorization": f"Bearer <OPENROUTER_API_KEY>"
  }
)

print(json.dumps(response.json(), indent=2))
```
To understand the OpenRouter API, please refer to the OpenRouter API docs at: https://openrouter.ai/docs/quickstart.