import requests,io

cookies = {
    'dvid': '5565f23c-cd0a-4882-b83b-41dfcab8860e',
    's_ecid': 'MCMID%7C12697088725315095381106968647157822291',
    'mmt-auth': 'MAT1cc2019af79c65212b53e17e14c31d63ada84444c968158b29b9ba2b0ecb2018326b75fa0167fb52b98cbbf3cbf411186P',
    'myBusinessSubscription': 'b2c',
    'MMT_LOYALTY': 'INACTIVE',
    '_gcl_au': '1.1.998913265.1704858180',
    '_fbp': 'fb.1.1704858248541.1651068423',
    'ccde': 'IN',
    'lang': 'eng',
    'isGdprRegion': '0',
    'bm_sz': '9C796AC0663A57CAE1CC367DB73195A8~YAAQLQkuF3WvhDiOAQAArKnuPRfrs1xHb0qD1sJda4A5a8VYxgrSAXFEIfD96letoZCYbuTqzjmsXh2Jx43Pjoz0H309b9+6gRYBBg6HDnaprbxztfY18g3amydjNBlkIDEkst4NeilWJxbNOyIIa+ByuaYLcXPpv63VkGtDes/Jdd9Nd7ImqgFFb8JosYULNJLOQGkkmsv/x3o+LqJp1dJDT57AsytRDwhCQVFlZTlfh3Jp60Axq1OgskL/2U43RkJZJZ5sSRuNxFkz1iY6uonwNTOmyceIohrYPfqamqTndvkklv2ol9hUUYMT1tqAB2PaSvIm/ywLfaYR48kdRDxhNk1Wlh/Eub5arheYJnJfsjuktXkrHPbAPYg=~3490117~4601400',
    'AMCVS_1E0D22CE527845790A490D4D%40AdobeOrg': '1',
    'AMCV_1E0D22CE527845790A490D4D%40AdobeOrg': '-1712354808%7CMCIDTS%7C19797%7CMCMID%7C12697088725315095381106968647157822291%7CMCAAMLH-1711040833%7C12%7CMCAAMB-1711040833%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1710443233s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C4.3.0',
    'bm_mi': '5666FC871F7E0FE16DED407C7ABFEAED~YAAQLQkuF47ThDiOAQAAi4XwPRcGxyZjAkR7DyY2XOvEeriJ4aoJb+AceoTTwRD2vdiubZzDVRaCYitVc05BP4o+dCRrXGA4pacjUVU3e0MN+Gx86fIy6I/9Rq4kBvdAV7KT5Dgiw997pOZ4KA1o8TGtCjBa0EtfyRZf8HUZGkJLIJYCmDsiN6fFCJTaMj+dLqCP90x/EFlTUoI2fWMQKVcSCrAS5Kz2Jgc/4tmf09mNkUNVSqrGghlMeBiuvxO5COV/XffgRml3S3KQ2DA6J1qFsUos2CiWkMYsboTVf1OaO5mjI544wx9IfL6WsOvar6W528ckIBjgoWvAr03f9B2YwAKJ~1',
    'mcid': '60bfdc9b-b566-4935-8a89-60a727cfffd8',
    'ak_bmsc': '957AA9879C11B9B602DE230C090865DE~000000000000000000000000000000~YAAQLQkuF/LThDiOAQAAyYnwPRfUnm8jp9gLBPr7eTUodp7Ccnbqnheliw+KTPQnYsch/FNgbY60xSPPAa/ktecBP2gxd+r/ldZdXcxsctC7Masj5Zu1lrwMy9qX9rY/u5YrLGzy8YP3eD1JnXG0Yylde56zl5vSMIhFvNDJgHv6MYyUCVtzp8e8pHE3b1AoOFFK1TfDoF2yrxtRVSn1TYoJYpVwQBqg3sE0pDt5DQiXglPEEr5FYxoYiHPhrHRmA8GEPA5nB3jydZv5VEbZV5CQ21PBlR2WyCKf512mgItVSWKK61EWqzry4pDsekNIYlsWDR0ydc51QJfZCDHLBlUFAAd6SKIh1yDgMkQHJoVY4i8BzbQMkhK6rFgH5PfaGQO/b4jQPD42XM5HkN35KadkazOs4dURj4ZQRHWLB9h3soaK57KwKiNUg7hfqblW22wm94yFKa0R8fhGo/NkgQdHCJ8UEb6jXv23IB051w0sz2IqqcNsvuStbT2VJZt8k1XR8Gzn',
    '_clck': '1x2vtrt%7C2%7Cfk2%7C0%7C1534',
    '_clsk': '18cgu8i%7C1710436945282%7C2%7C1%7Ci.clarity.ms%2Fcollect',
    '_adck_id': 'c425d9a7c8f1233fdcfda87659a73f81c1edc530617eefcfd84565b6c981010e',
    'bm_sv': 'F4ED9C50C5EB61D63BD4EE17A7A5BFB8~YAAQRvY3F9eFwDWOAQAA0Mb8PRd32oBPYsmyKLIm4GLvjBRwpMG2SIzmEj5JlXzap929RtOa7y0s9lJqL2BE94Yi5GTmkAqFWzSQcRiHs0tT5ID8uTJOlnsaUQf4VfEBS82vUGpC4sq8UcCUap1PIvYmOQU0rROfTtxZHpP0jDCOu0Q4fz+963+0BEXm86sY89aQ1LsRyi3Xc2R4L9si+MUWK95Re+/E1SjwQ+Tc+FHiNl7IHdxU3ToISQ3EUylOH1b77w==~1',
    'visitNumber': '%7B%22number%22%3A1%2C%22time%22%3A1710437052518%7D',
    'MMYTUUID': '24617072-3124-684d-3863-35624d572465.1710437055043728',
    '_abck': 'C022053A002078CF33D345EF4DB2D296~0~YAAQRvY3F5ucwDWOAQAA/kP+PQt5BPdcJ/1HUCe97SHuigT/WSGABgufl88S8n6D7YfNAnGGB+sdjbhFxATPvxr99PIXzFffTkXlwaOzbOU1b5RJOoAK384YFGN7INZwup6men646PDtpb9mCRApb0scjSS9o0ClC319OPB5vBBS25vu31BkDM9MtW+6vYub1gAbr5TxpfR89xB+X3nfoam6cExuHRkMJ8pSrPf0gRtBnbFFOnc4z1p5Qai0AcDLCkNnKr/CxOs/BNjT28/adlHHPECjH5jT6OogQkpfGlYlI3srOUZRQglfmlrBKWV9Y1mpzqTXtQV1z78vK6OiwPKSwGdrfsfEghrWErvE/7jwld65AKNLh57uG/jS22WFdBHOB3qiFN1fnUTC/fF71HlFVcyxGYkT0rXPTw==~-1~-1~-1',
    's_pers': '%20s_vnum%3D1711909800661%2526vn%253D1%7C1711909800661%3B%20s_depth%3D6%7C1710438849558%3B%20s_lv%3D1710437063414%7C1805045063414%3B%20s_lv_s%3DMore%2520than%25207%2520days%7C1710438863414%3B%20gpv_pn%3Dfunnel%253Adomestic%2520hotels%253Ahoteldetails%7C1710438863415%3B%20s_invisit%3Dtrue%7C1710438863416%3B%20s_nr3650%3D1710437063416-Repeat%7C2025797063416%3B%20s_nr30%3D1710437063417-Repeat%7C1713029063417%3B%20s_nr120%3D1710437063417-Repeat%7C1720805063417%3B%20s_nr7%3D1710437063418-New%7C1711041863418%3B',
    's_sess': '%20s_cc%3Dtrue%3B%20tp%3D11252%3B%20s_ppv%3Dfunnel%25253Adomestic%252520hotels%25253Ahoteldetails%252C98%252C67%252C10995%3B%20s_sq%3Dmmtprod%253D%252526c.%252526a.%252526activitymap.%252526page%25253Dfunnel%2525253Adomestic%25252520hotels%2525253Ahoteldetails%252526link%25253D2%252526region%25253Ddetpg_review_ratings_pagination%252526pageIDType%25253D1%252526.activitymap%252526.a%252526.c%252526pid%25253Dfunnel%2525253Adomestic%25252520hotels%2525253Ahoteldetails%252526pidt%25253D1%252526oid%25253Dhttps%2525253A%2525252F%2525252Fwww.makemytrip.com%2525252Fhotels%2525252Fhotel-details%2525252F%2525253FhotelId%2525253D200709211000561392%25252526_uCurrency%2525253DINR%25252526checkin%2525253D0%252526ot%25253DA%3B',
}

headers = {
    'authority': 'mapi.makemytrip.com',
    'accept': 'application/json',
    'accept-language': 'en-GB,en-US;q=0.9,en;q=0.8',
    'content-type': 'application/json',
    # 'cookie': 'dvid=5565f23c-cd0a-4882-b83b-41dfcab8860e; s_ecid=MCMID%7C12697088725315095381106968647157822291; mmt-auth=MAT1cc2019af79c65212b53e17e14c31d63ada84444c968158b29b9ba2b0ecb2018326b75fa0167fb52b98cbbf3cbf411186P; myBusinessSubscription=b2c; MMT_LOYALTY=INACTIVE; _gcl_au=1.1.998913265.1704858180; _fbp=fb.1.1704858248541.1651068423; ccde=IN; lang=eng; isGdprRegion=0; bm_sz=9C796AC0663A57CAE1CC367DB73195A8~YAAQLQkuF3WvhDiOAQAArKnuPRfrs1xHb0qD1sJda4A5a8VYxgrSAXFEIfD96letoZCYbuTqzjmsXh2Jx43Pjoz0H309b9+6gRYBBg6HDnaprbxztfY18g3amydjNBlkIDEkst4NeilWJxbNOyIIa+ByuaYLcXPpv63VkGtDes/Jdd9Nd7ImqgFFb8JosYULNJLOQGkkmsv/x3o+LqJp1dJDT57AsytRDwhCQVFlZTlfh3Jp60Axq1OgskL/2U43RkJZJZ5sSRuNxFkz1iY6uonwNTOmyceIohrYPfqamqTndvkklv2ol9hUUYMT1tqAB2PaSvIm/ywLfaYR48kdRDxhNk1Wlh/Eub5arheYJnJfsjuktXkrHPbAPYg=~3490117~4601400; AMCVS_1E0D22CE527845790A490D4D%40AdobeOrg=1; AMCV_1E0D22CE527845790A490D4D%40AdobeOrg=-1712354808%7CMCIDTS%7C19797%7CMCMID%7C12697088725315095381106968647157822291%7CMCAAMLH-1711040833%7C12%7CMCAAMB-1711040833%7CRKhpRz8krg2tLO6pguXWp5olkAcUniQYPHaMWWgdJ3xzPWQmdj0y%7CMCOPTOUT-1710443233s%7CNONE%7CMCAID%7CNONE%7CvVersion%7C4.3.0; bm_mi=5666FC871F7E0FE16DED407C7ABFEAED~YAAQLQkuF47ThDiOAQAAi4XwPRcGxyZjAkR7DyY2XOvEeriJ4aoJb+AceoTTwRD2vdiubZzDVRaCYitVc05BP4o+dCRrXGA4pacjUVU3e0MN+Gx86fIy6I/9Rq4kBvdAV7KT5Dgiw997pOZ4KA1o8TGtCjBa0EtfyRZf8HUZGkJLIJYCmDsiN6fFCJTaMj+dLqCP90x/EFlTUoI2fWMQKVcSCrAS5Kz2Jgc/4tmf09mNkUNVSqrGghlMeBiuvxO5COV/XffgRml3S3KQ2DA6J1qFsUos2CiWkMYsboTVf1OaO5mjI544wx9IfL6WsOvar6W528ckIBjgoWvAr03f9B2YwAKJ~1; mcid=60bfdc9b-b566-4935-8a89-60a727cfffd8; ak_bmsc=957AA9879C11B9B602DE230C090865DE~000000000000000000000000000000~YAAQLQkuF/LThDiOAQAAyYnwPRfUnm8jp9gLBPr7eTUodp7Ccnbqnheliw+KTPQnYsch/FNgbY60xSPPAa/ktecBP2gxd+r/ldZdXcxsctC7Masj5Zu1lrwMy9qX9rY/u5YrLGzy8YP3eD1JnXG0Yylde56zl5vSMIhFvNDJgHv6MYyUCVtzp8e8pHE3b1AoOFFK1TfDoF2yrxtRVSn1TYoJYpVwQBqg3sE0pDt5DQiXglPEEr5FYxoYiHPhrHRmA8GEPA5nB3jydZv5VEbZV5CQ21PBlR2WyCKf512mgItVSWKK61EWqzry4pDsekNIYlsWDR0ydc51QJfZCDHLBlUFAAd6SKIh1yDgMkQHJoVY4i8BzbQMkhK6rFgH5PfaGQO/b4jQPD42XM5HkN35KadkazOs4dURj4ZQRHWLB9h3soaK57KwKiNUg7hfqblW22wm94yFKa0R8fhGo/NkgQdHCJ8UEb6jXv23IB051w0sz2IqqcNsvuStbT2VJZt8k1XR8Gzn; _clck=1x2vtrt%7C2%7Cfk2%7C0%7C1534; _clsk=18cgu8i%7C1710436945282%7C2%7C1%7Ci.clarity.ms%2Fcollect; _adck_id=c425d9a7c8f1233fdcfda87659a73f81c1edc530617eefcfd84565b6c981010e; bm_sv=F4ED9C50C5EB61D63BD4EE17A7A5BFB8~YAAQRvY3F9eFwDWOAQAA0Mb8PRd32oBPYsmyKLIm4GLvjBRwpMG2SIzmEj5JlXzap929RtOa7y0s9lJqL2BE94Yi5GTmkAqFWzSQcRiHs0tT5ID8uTJOlnsaUQf4VfEBS82vUGpC4sq8UcCUap1PIvYmOQU0rROfTtxZHpP0jDCOu0Q4fz+963+0BEXm86sY89aQ1LsRyi3Xc2R4L9si+MUWK95Re+/E1SjwQ+Tc+FHiNl7IHdxU3ToISQ3EUylOH1b77w==~1; visitNumber=%7B%22number%22%3A1%2C%22time%22%3A1710437052518%7D; MMYTUUID=24617072-3124-684d-3863-35624d572465.1710437055043728; _abck=C022053A002078CF33D345EF4DB2D296~0~YAAQRvY3F5ucwDWOAQAA/kP+PQt5BPdcJ/1HUCe97SHuigT/WSGABgufl88S8n6D7YfNAnGGB+sdjbhFxATPvxr99PIXzFffTkXlwaOzbOU1b5RJOoAK384YFGN7INZwup6men646PDtpb9mCRApb0scjSS9o0ClC319OPB5vBBS25vu31BkDM9MtW+6vYub1gAbr5TxpfR89xB+X3nfoam6cExuHRkMJ8pSrPf0gRtBnbFFOnc4z1p5Qai0AcDLCkNnKr/CxOs/BNjT28/adlHHPECjH5jT6OogQkpfGlYlI3srOUZRQglfmlrBKWV9Y1mpzqTXtQV1z78vK6OiwPKSwGdrfsfEghrWErvE/7jwld65AKNLh57uG/jS22WFdBHOB3qiFN1fnUTC/fF71HlFVcyxGYkT0rXPTw==~-1~-1~-1; s_pers=%20s_vnum%3D1711909800661%2526vn%253D1%7C1711909800661%3B%20s_depth%3D6%7C1710438849558%3B%20s_lv%3D1710437063414%7C1805045063414%3B%20s_lv_s%3DMore%2520than%25207%2520days%7C1710438863414%3B%20gpv_pn%3Dfunnel%253Adomestic%2520hotels%253Ahoteldetails%7C1710438863415%3B%20s_invisit%3Dtrue%7C1710438863416%3B%20s_nr3650%3D1710437063416-Repeat%7C2025797063416%3B%20s_nr30%3D1710437063417-Repeat%7C1713029063417%3B%20s_nr120%3D1710437063417-Repeat%7C1720805063417%3B%20s_nr7%3D1710437063418-New%7C1711041863418%3B; s_sess=%20s_cc%3Dtrue%3B%20tp%3D11252%3B%20s_ppv%3Dfunnel%25253Adomestic%252520hotels%25253Ahoteldetails%252C98%252C67%252C10995%3B%20s_sq%3Dmmtprod%253D%252526c.%252526a.%252526activitymap.%252526page%25253Dfunnel%2525253Adomestic%25252520hotels%2525253Ahoteldetails%252526link%25253D2%252526region%25253Ddetpg_review_ratings_pagination%252526pageIDType%25253D1%252526.activitymap%252526.a%252526.c%252526pid%25253Dfunnel%2525253Adomestic%25252520hotels%2525253Ahoteldetails%252526pidt%25253D1%252526oid%25253Dhttps%2525253A%2525252F%2525252Fwww.makemytrip.com%2525252Fhotels%2525252Fhotel-details%2525252F%2525253FhotelId%2525253D200709211000561392%25252526_uCurrency%2525253DINR%25252526checkin%2525253D0%252526ot%25253DA%3B',
    'currency': 'INR',
    'language': 'eng',
    'mmt-auth': 'MAT1cc2019af79c65212b53e17e14c31d63ada84444c968158b29b9ba2b0ecb2018326b75fa0167fb52b98cbbf3cbf411186P',
    'origin': 'https://www.makemytrip.com',
    'os': 'desktop',
    'referer': 'https://www.makemytrip.com/',
    'region': 'in',
    'sec-ch-ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-site',
    'server': 'b2c',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'usr-mcid': '12697088725315095381106968647157822291',
    'vid': '60bfdc9b-b566-4935-8a89-60a727cfffd8',
    'visitor-id': '60bfdc9b-b566-4935-8a89-60a727cfffd8',
}

params = {
    'srcClient': 'DESKTOP',
    'contextType': 'null',
    'language': 'eng',
    'region': 'in',
    'currency': 'INR',
    'idContext': 'B2C',
    'countryCode': 'IN',
}

json_data = {
    'filter': {
        'ota': 'MMT',
        'travelTagTypes': [],
    },
    'sortCriteria': {
        'sortBy': 'Latest first',
    },
    'start': 1,
    'limit': 1830,
    'bookingDevice': 'DESKTOP',
}

response = requests.post(
    'https://mapi.makemytrip.com/clientbackend/entity/api/hotel/200709211000561392/flyfishReviews',
    params=params,
    cookies=cookies,
    headers=headers,
    json=json_data,
)

# Note: json_data will not be serialized by requests
# exactly as it was in the original request.
#data = '{"filter":{"ota":"MMT","travelTagTypes":[]},"sortCriteria":{"sortBy":"Latest first"},"start":5,"limit":5,"bookingDevice":"DESKTOP"}'
#response = requests.post(
#    'https://mapi.makemytrip.com/clientbackend/entity/api/hotel/200709211000561392/flyfishReviews',
#    params=params,
#    cookies=cookies,
#    headers=headers,
#    data=data,
#)

data=response.json()
data=data["payload"]
data=data["response"]
rows=data["MMT"]
print(len(rows))
with io.open("review.txt","a",encoding="utf-8")as f1:
    for row in rows:
        f1.write(row["reviewText"]+"\n\n")
f1.close()