import requests
from bs4 import BeautifulSoup
import os

base_url = 'https://tspace.library.utoronto.ca'
url = 'https://tspace.library.utoronto.ca/handle/1807/24487'

page = requests.get(url)

soup = BeautifulSoup(page.text)

if not os.path.exists('TESS'):
    os.mkdir('TESS')

table = soup.find_all('table')[0]
len(soup.find_all('table'))

len(table.find_all('tr'))


def download_files2(url, directory):
    print(url)
    page = requests.get(url)
    soup = BeautifulSoup(page.text)
    table = soup.find_all('div', {'class': "item-files"})[0].ul
    for row in table.find_all('li'):
        print(row.a['href'])
        print(row.string)
        with open(directory + '/' + row.string, 'wb') as handle:
            handle.write(requests.get(base_url + row.a['href']).content)
        # return


for row in table.find_all('tr')[1:]:
    folder_name = 'TESS/' + row.find_all('td')[1].string
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    download_files2(base_url + row.find_all('td')[1].a['href'], folder_name)
    # break
