import os
import time
import pandas as pd
from nordvpn_switcher import initialize_VPN,rotate_VPN,terminate_VPN
settings = initialize_VPN()
rotate_VPN(settings)
from auxiliary.unitedstateszipcode_scraper import ZipCodeUSA
zipCodeScraper = ZipCodeUSA()

donors = pd.read_csv("D:\Programming\Python\DonorsChoose\data\DonorsChoose\Donors.csv")
#donors = donors.loc[donors["Donor State"] != 'other']

zipcodes = pd.Series(donors["Donor Zip"].unique(), name="zipcodes")
zipcodes = pd.to_numeric(zipcodes, errors= "coerce").dropna().astype(int).astype(str)
for i in range(1, 3):
    zipcodes.loc[zipcodes.str.len() == i] = (zipcodes.loc[zipcodes.str.len() == i] + "0"*(3-i)).values
zipcodes = zipcodes.unique()

zipcode_df = pd.DataFrame(index=range(len(zipcodes)), columns=["id", "Population", "Population Density", "Housing Units", "Median Home Value", "Land Area", "Water Area", "Occupied Housing Units", "Median Household Income"])

additions = ["01", "02", "05",  "00", "10", "20", "30", "40", "50", "60", "70", "80", "90", "21", "43", "24", "99"]
max_tries = len(additions)
errors = 0
loops = 0

for i, item in enumerate(zipcodes):
    tries = 0
    doc = pd.Series()
    while len(doc) < 2 and max_tries > tries:
        time.sleep(0.2)
        zipcode = item + additions[tries]
        doc = pd.Series(zipCodeScraper.scrape_results(zipcode))
        doc = doc.str.replace("$", "")
        doc = doc.str.replace(",", "")
        doc = doc.str.replace("n/a", "")
        doc = doc.drop("zipcode")
        if sum(pd.to_numeric(doc.drop("id"), errors= "coerce").dropna()) > 0:
            zipcode_df.loc[i] = doc
        tries += 1
        loops += 1
        if loops % 100 == 0:
            print("Switching Connection")
            rotate_VPN(settings)
            time.sleep(20)

    if tries >= max_tries:
        print("{}/{}   Error for id {}".format(i, len(zipcodes), item))
        errors += 1
    else:
        print("{}/{}".format(i, len(zipcodes)))

print("{} Errors in total".format(errors))
terminate_VPN(settings)

zipcode_df.to_csv(os.path.join("D:\Programming\Python\DonorsChoose\data\EconomicIndicators", "ZipCodes_AreaContext.csv"), index=False)