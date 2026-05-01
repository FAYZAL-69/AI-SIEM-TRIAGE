from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(model="llama3.2:1b", temperature=0.4, num_predict=150)

def generate_report(alert_row, similar_alerts):
    prompt = f"""
    You are a senior SOC analyst. Here is one alert:
    {alert_row.to_dict()}
    
    Here are 2-3 similar alerts that might be correlated:
    {similar_alerts}
    
    Write a short plain-English investigation report:
    - Is this a real threat or false positive?
    - Map to MITRE ATT&CK if possible
    - Suggest next 2 steps
    Keep it under 150 words.
    """
    messages = [SystemMessage(content="You are an expert SOC analyst."), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    return response.content