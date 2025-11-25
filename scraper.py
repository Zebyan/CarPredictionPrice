import requests
from bs4 import BeautifulSoup
import csv
import time
import logging
import os
from collections import defaultdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OLXRomaniaScraper:
    def __init__(self, output_dir='carData'):
        self.output_dir = output_dir  # Directory to save CSVs
        self.headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            )
        }
        self.base_url = 'https://www.olx.ro/auto-masini-moto-ambarcatiuni/autoturisme/'
        self.base_params = {'currency': 'EUR'}
        self.rows = []
        self.columns = [
            'marca', 'model', 'pret', 'capacitate motor', 'putere',
            'combustibil', 'caroserie', 'rulaj', 'culoare',
            'an fabricatie', 'cutie viteza','descriere'
        ]
        
        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def scrape_page(self, url, params=None):
        try:
            logger.info(f"Scraping: {url}")
            resp = requests.get(url, headers=self.headers, timeout=15, params=params)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return None

    def collect_listing_urls(self, soup):
        urls = set()
        for a in soup.select('a[href*="/d/oferta/"]'):
            href = a.get('href')
            if not href:
                continue
            if href.startswith('//'):
                href = 'https:' + href
            elif href.startswith('/'):
                href = 'https://www.olx.ro' + href
            urls.add(href)
        logger.info(f"Collected {len(urls)} detail URLs from listing page")
        return list(urls)

    def extract_car_detail_row(self, soup, url, brand):
        """Extract car details into the requested fields"""
        row = {col: 'N/A' for col in self.columns}

        # Asign marca based on the current brand
        row['marca'] = brand


        # --- Price ---
        price_elem = soup.find('div', {'data-testid': 'ad-price-container'})
        if price_elem:
            h3 = price_elem.find('h3')
            if h3:
                row['pret'] = h3.get_text(strip=True)

        # --- Parameters ---
        params_container = soup.find('div', {'data-testid': 'ad-parameters-container'})
        if params_container:
            for p in params_container.find_all('p'):
                text = p.get_text(" ", strip=True)
                if ':' in text:
                    label, value = map(str.strip, text.split(':', 1))
                    label = label.lower()
                    if 'model' in label:
                        row['model'] = value
                    elif 'capacitate motor' in label:
                        row['capacitate motor'] = value
                    elif 'putere' in label:
                        row['putere'] = value
                    elif 'combustibil' in label:
                        row['combustibil'] = value
                    elif 'caroserie' in label:
                        row['caroserie'] = value
                    elif 'rulaj' in label:
                        row['rulaj'] = value
                    elif 'culoare' in label:
                        row['culoare'] = value
                    elif 'an de fabricatie' in label:
                        row['an fabricatie'] = value
                    elif 'cutie de viteze' in label:
                        row['cutie viteza'] = value
        description_container = soup.find('div',{'data-cy' : 'ad_description'})
        if description_container:
            description_div = description_container.find('div', class_='css-19duwlz')
            if description_div:
                row['descriere'] = description_div.get_text("\n", strip=True)

        return row

    def save_to_csv(self):
        if not self.rows:
            logger.warning("No cars to save")
            return
        
        # Group rows by marca
        marca_groups = defaultdict(list)
        for row in self.rows:
            marca_groups[row['marca']].append(row)
        
        # Save separate CSV for each marca
        for marca, rows in marca_groups.items():
            filename = f'{self.output_dir}/cars_{marca.lower().replace(" ", "_")}.csv'
            try:
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self.columns)
                    writer.writeheader()
                    writer.writerows(rows)
                logger.info(f"Saved {len(rows)} cars for {marca} to {filename}")
            except Exception as e:
                logger.error(f"Failed to save CSV for {marca}: {e}")

    def run(self, num_pages=20, brands=['audi', 'bmw', 'chevrolet', 'citroen', 'dacia','fiat','ford','honda','hyundai','kia','mazda','mercedes-benz','mitsubishi','nissan','opel','peugeot','porche','renault','seat','skoda','suzuki','tesla','toyota','volkswagen','volvo']):
        # Loop through the list of brands
        for brand in brands:
            detail_urls = []  # Reset detail_urls for each brand
            
            # Scrape multiple pages for each brand
            for page in range(1, num_pages + 1):
                params = dict(self.base_params)
                params['page'] = page
                url = f"{self.base_url}{brand}"  # Generate URL for the specific brand
                soup = self.scrape_page(url, params=params)
                if not soup:
                    break
                urls = self.collect_listing_urls(soup)
                detail_urls.extend(urls)
                time.sleep(2)  # Sleep for 2 seconds to avoid overloading the server

            detail_urls = list(dict.fromkeys(detail_urls))  # Deduplicate URLs
            logger.info(f"Total detail pages to scrape for {brand}: {len(detail_urls)}")

            # Scrape the car listings for the current brand
            for idx, u in enumerate(detail_urls, start=1):
                dsoup = self.scrape_page(u)
                if not dsoup:
                    continue
                row = self.extract_car_detail_row(dsoup, u, brand)
                self.rows.append(row)
                logger.info(f"[{idx}/{len(detail_urls)}] Scraped: {row.get('marca')} {row.get('model')} - {row.get('pret')}")
                time.sleep(2)  # Sleep to ensure the scraper doesn't overwhelm the server

        # Save the data after scraping all the brands
        self.save_to_csv()
        logger.info("Done.")

if __name__ == '__main__':
    scraper = OLXRomaniaScraper(output_dir='carData')  # Save data in 'carData' directory
    scraper.run(num_pages=20)  # Scrape 5 pages for each brand
