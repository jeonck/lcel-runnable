# https://wikidocs.net/235886
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import StreamingStdOutCallbackHandler
import os
import dotenv

dotenv.load_dotenv()

# OpenAI API 키 설정 (환경 변수에서 가져오기)
openai_api_key = os.getenv("OPENAI_API_KEY")

# OpenAI LLM 초기화
# llm = ChatOpenAI(api_key=openai_api_key)
llm = ChatOpenAI(
    base_url="https://api.openai.com/v1",  # Updated to the proper API endpoint
    api_key=openai_api_key,  # Replace with your actual API key
    model="gpt-4o",
    # streaming=True,
    # callbacks=[StreamingStdOutCallbackHandler()]
)
# 프롬프트 템플릿 정의
prompt_template = PromptTemplate.from_template(
    """
    Q: {question}
    A:
    """
)

# Runnable 사용하여 체인 구성
complete_chain = (
    {"question": RunnablePassthrough()}  # 질문을 그대로 전달
    | prompt_template  # 프롬프트 템플릿 적용
    | llm
    | StrOutputParser()  # 문자열 출력 파서 적용
)

# 질문 입력 예제
question = "What is the capital of France?"

# 체인 실행
response = complete_chain.invoke({"question": question})
print(response)
