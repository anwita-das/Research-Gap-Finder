from src.gap_finding.llm_helper import LLMHelper

llm = LLMHelper()

response = llm.generate(
    "Say hello in exactly five words."
)

print(response)