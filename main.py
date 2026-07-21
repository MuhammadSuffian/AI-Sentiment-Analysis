from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os
import time

# Streamlit app URL from environment variable (or default)
STREAMLIT_URL = os.environ.get(
    "STREAMLIT_APP_URL",
    "https://ai-sentiment-analysisgit.streamlit.app/"
)

# How long to keep the page open (seconds) so Streamlit registers real activity
VISIT_DURATION = 20

def main():
    options = Options()
    options.add_argument('--headless=new')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        driver.get(STREAMLIT_URL)
        print(f"Opened {STREAMLIT_URL}")

        time.sleep(VISIT_DURATION)
        print(f"Kept page open for {VISIT_DURATION}s ✅")

    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)
    finally:
        driver.quit()
        print("Script finished.")

if __name__ == "__main__":
    main()
