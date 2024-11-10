import requests


class HackerNewsCollector:
    def __init__(self):
        self.base_url = "https://hacker-news.firebaseio.com/v0/"

    def fetch_top_stories(self, limit=10):
        """En popüler hikayeleri al."""
        top_stories_url = self.base_url + "topstories.json?print=pretty"
        top_stories = requests.get(top_stories_url).json()[:limit]

        stories_details = []
        for story_id in top_stories:
            # Hikaye detaylarını al
            story_url = self.base_url + f"item/{story_id}.json?print=pretty"
            story_details = requests.get(story_url).json()
            stories_details.append(story_details)

        return stories_details
