import os
import stardog

from dotenv import load_dotenv

load_dotenv()

SD_USER = os.getenv('SD_USER')
SD_PASSWORD = os.getenv('SD_PASSWORD')

def send_stardog_consult(query):
    conn_details = {
    'endpoint': 'https://sd-0d6bd678.stardog.cloud:5820',
    'username': SD_USER,
    'password': SD_PASSWORD
    }

    # Initialize connection
    with stardog.Connection('PD_KG', **conn_details) as conn:        
        # Execute the query
        if 'CONSTRUCT' in query:
            results = conn.graph(query)
        else:
            results = conn.select(query)

    return results