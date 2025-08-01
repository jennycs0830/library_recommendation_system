template = """
書名: {bi_title}
作者: {bi_auther}
大綱: {bi_content}
"""

param_dict = ["bi_title", "bi_auther", "bi_content"]

def build_book_text(row):
    return template.format(**{key: row.get(key, "") for key in param_dict})
