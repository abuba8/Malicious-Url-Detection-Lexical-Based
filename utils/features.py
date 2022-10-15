import pandas as pd
from urllib.parse import urlparse
from tld import get_tld
from helper import fd_length, tld_length, digit_count, letter_count, \
    no_of_dir, having_ip_address, shortening_service


print('Loading Data...')
df1 = pd.read_csv("./data/finaldata1.csv")
df2 = pd.read_csv("./data/finaldata2.csv")

frames = [df1, df2]
urldata = pd.concat(frames)

print('Extracting Url Length...')
urldata['url_length'] = urldata['url'].apply(lambda i: len(str(i)))

print('Extracting hostname Length...')
urldata['hostname_length'] = urldata['url'].apply(lambda i: len(urlparse(i).netloc))

print('Extracting path Length...')
urldata['path_length'] = urldata['url'].apply(lambda i: len(urlparse(i).path))

print('Extracting full directory Length...')
urldata['fd_length'] = urldata['url'].apply(lambda i: fd_length(i))

print('Extracting top level domain Length...')
urldata['tld'] = urldata['url'].apply(lambda i: get_tld(i,fail_silently=True))
urldata['tld_length'] = urldata['tld'].apply(lambda i: tld_length(i))
urldata = urldata.drop("tld",1)

print('Extracting count features...')
urldata['count-'] = urldata['url'].apply(lambda i: i.count('-'))
urldata['count@'] = urldata['url'].apply(lambda i: i.count('@'))
urldata['count?'] = urldata['url'].apply(lambda i: i.count('?'))
urldata['count%'] = urldata['url'].apply(lambda i: i.count('%'))
urldata['count.'] = urldata['url'].apply(lambda i: i.count('.'))
urldata['count='] = urldata['url'].apply(lambda i: i.count('='))
urldata['count-http'] = urldata['url'].apply(lambda i : i.count('http'))
urldata['count-https'] = urldata['url'].apply(lambda i : i.count('https'))
urldata['count-www'] = urldata['url'].apply(lambda i: i.count('www'))
urldata['count-digits']= urldata['url'].apply(lambda i: digit_count(i))
urldata['count-letters']= urldata['url'].apply(lambda i: letter_count(i))
urldata['count_dir'] = urldata['url'].apply(lambda i: no_of_dir(i))


print('Extracting feature use of ip...')
urldata['use_of_ip'] = urldata['url'].apply(lambda i: having_ip_address(i))

print('shortened url feature...')
urldata['short_url'] = urldata['url'].apply(lambda i: shortening_service(i))

print('Feature extraction completed, saving dataframe')
urldata.to_csv('./data/extracted_features.csv')
