"""
python script to scrape the results from unitedstateszipcodes and save to a file
"""

from bs4 import BeautifulSoup
import requests, json, ast

class ZipCodeUSA:
	def __init__(self):
		self.base_url = "https://www.unitedstateszipcodes.org"
		self.headers = {
			'accept':'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
			'accept-encoding':'gzip, deflate, br',
			'accept-language':'en-GB,en-US;q=0.8,en;q=0.6',
			'cache-control':'max-age=0',
			'upgrade-insecure-requests':'1',
			'user-agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36'
		}

	def scrape_results(self, zipcode):
		url = self.base_url + "/" + zipcode + "/"
		resp = requests.get(url, headers = self.headers)
		corn = BeautifulSoup(resp.text, features="html.parser")
		doc = {'zipcode' : zipcode, 'id' : zipcode[:3]}
		for tab in corn.findAll('table'):
			if "Population Density" in str(tab):
				heads = tab.findAll('th')
				vals = tab.findAll('td', {'class' : 'text-right'})
				for i, head in enumerate(heads):
					doc[head.text] = vals[i].text
			if "Land Area" in str(tab):
				heads = tab.findAll('th')
				vals = tab.findAll('td', {'class' : 'text-right'})
				for i, head in enumerate(heads):
					doc[head.text] = vals[i].text
		return doc

if __name__ == '__main__':

	### Get Scraped Results and Store in a file
	fout = open("scraped_results.txt", "w")
	
	header = [u'Housing Units', 'zipcode', u'Water Area', u'Median Home Value', u'Median Household Income', u'Population Density', u'Occupied Housing Units', u'Population', 'id', u'Land Area']
	fout.write("\t".join(header) + "\n")
	
	list_of_zipcodes = ["33063","33064"]
	for zipcode in list_of_zipcodes:
		try:
			doc = ZipCodeUSA().scrape_results(zipcode)
			
			doc['_id'] = zipcode
			if len(doc.keys()) == 2:
				continue

			row = []
			for h in header:
				if h in doc:
					row.append(doc[h].replace("$","").replace(",","").replace("n/a",""))
				else:
					row.append("0")
			fout.write("\t".join(row) + "\n")
		except Exception as E:
			print("Exception Occured", E, zipcode)
			continue
	fout.close()