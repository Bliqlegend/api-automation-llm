# summary = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {
#             "role": "user",
#             "content": summary_prompt.format(
#                 passage=truncate(encoding_gpt3, doc.page_content, 3100)
#             ),
#         }
#     ],
#     max_tokens=700,
# )["choices"][0]["message"]["content"]


# summary_template = open("assets/prompt_summary.txt").read() + "\n"
