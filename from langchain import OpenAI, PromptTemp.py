from langchain import OpenAI, PromptTemplate
from langchain.chains import LLMChain, LLMMathChain, TransformChain, SequentialChain, SimpleSequentialChain

# This is an LLMChain to write a synopsis given a title of a play.
playwright_llm = OpenAI(temperature=.9, openai_api_key = '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')

playwright_template = """

You are a playwright. Given the title of play, write a synopsis for that title.
Your style is witty, humorous, light-hearted. All your plays are written using
concise language, to the point, and are brief.

Title: {title}

Playwright: This is a synopsis for the above play:

"""

playwright_prompt_template = PromptTemplate(input_variables=["title"], template=playwright_template)

synopsis_chain = LLMChain(llm=playwright_llm, prompt=playwright_prompt_template)

# This is an LLMChain to write a review of a play given a synopsis.
critic_llm = OpenAI(temperature=.5, openai_api_key = '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')

synopsis_template = """

You are a play critic from the New York Times.

Given the synopsis of play, it is your job to write a review for that play.
You're the Simon Cowell of play critics and always deliver scathing reviews.

Play Synopsis: {synopsis}

Review from a New York Times play critic of the above play:"""

critic_prompt_template = PromptTemplate(input_variables=["synopsis"], template=synopsis_template)

review_chain = LLMChain(llm=critic_llm, prompt=critic_prompt_template)

# This is the overall chain where we run these two chains in sequence.

overall_chain = SimpleSequentialChain(chains= [synopsis_chain, review_chain],
                                      verbose=True)


play_title = "Abigail Aryan and the Motley Crew of Multi-Agent Systems"

review = overall_chain.run(play_title0