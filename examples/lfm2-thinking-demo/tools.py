# import os
# from typing import Any




def calculator(mathematical_expression: str) -> float:
    """Evaluates a mathematical expression
        
    Args:
        mathematical_expression: Mathematical expression to evaluate

    Returns:
        the result of the evaluation of the mathematical expression
    """
    expression = mathematical_expression
    try:
        # Safety: only allow basic math operations
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Invalid characters in expression"
        return eval(expression)
    except Exception as e:
        return f"Error: {str(e)}"

tools = {
    'calculator': calculator,
}



# from tavily import TavilyClient

# api_key = os.getenv('TAVILY_API_KEY')
# client = TavilyClient(api_key=api_key)

# def search_product_updates(
#     company_name: str,
#     domains: list[str],
# ) -> list[Any]:
#     """Search for product updates from a company.
    
#     Args:
#         company_name: Company to search for
#         domains: Company domains for self-reported news
    
#     Returns:
#         List of results with 'search_type' field indicating source
#     """
#     all_results = []

#     # Self-reported news from company domains
#     company_results = client.search(
#         query=f"{company_name} product news, updates, releases, and announcements",
#         search_depth="basic",
#         max_results=10,
#         include_domains=domains,
#     )

#     for result in company_results["results"]:
#         result["search_type"] = "Self-reported News"
#         all_results.append(result)

#     # Third-party coverage from news sources
#     news_results = client.search(
#         query=f"{company_name} product news, updates, releases, and announcements",
#         search_depth="basic",
#         max_results=10,
#         time_range="month",
#         topic="news",
#     )

#     for result in news_results["results"]:
#         result["search_type"] = "Third-party Coverage"
#         all_results.append(result)

#     return all_results

# # maps tool names to tool callers
# TOOLS = {
#     "search_product_updates": search_product_updates,
# }