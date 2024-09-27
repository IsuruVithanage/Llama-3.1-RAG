from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Define the prompt template for the LLM (no documents needed)
prompt = PromptTemplate(
    template="""
    Question: {question}
    Answer:
    """,
    input_variables=["question"],
)

# Initialize the LLM with Llama 3.1 model, pointing to your local instance
llm = ChatOllama(
    model="llama3.1",  # Point this to your local running instance of llama3.1
    temperature=0,  # Set temperature to control randomness
)

# Create a chain combining the prompt template and LLM
chain = prompt | llm | StrOutputParser()


# Define a class to use the model for direct question answering
class SimpleLLMApplication:
    def __init__(self, chain):
        self.chain = chain

    def run(self, question):
        # Directly query the model
        answer = self.chain.invoke({"question": question})
        return answer


# Initialize the application
app = SimpleLLMApplication(chain)

# Example usage
question = "My Gender: Male, Age: 45, Chol: 220 mg/dL, HDL: 40 mg/dL, LDL: 150 mg/dL, TG 180 mg/dL, BMI: 28 analyze this data recommend me a diet?"
answer = app.run(question)

print("Question:", question)
print("Answer:", answer)
