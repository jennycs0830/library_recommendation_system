def create_pagination(total_pages, current_page, on_page_change):
    pagination = []
    
    if current_page > 0:
        pagination.append({"label": "⬅️ 上一頁", "page": current_page - 1})

    for page in range(total_pages):
        pagination.append({"label": str(page + 1), "page": page})

    if current_page < total_pages - 1:
        pagination.append({"label": "➡️ 下一頁", "page": current_page + 1})

    return pagination

def render_pagination(pagination, current_page, on_page_change):
    for item in pagination:
        if item['page'] == current_page:
            print(f"[{item['label']}]")  # Current page
        else:
            print(f"{item['label']}")  # Other pages
