from langchain_community.document_loaders import AsyncChromiumLoader
from bs4 import BeautifulSoup
import io

urls=["https://www.espncricinfo.com/records/most-runs-in-career-223646"]
loader=AsyncChromiumLoader(urls)
html=loader.load()
data=[]
soup=BeautifulSoup(html[0].page_content,"html.parser")
try:
    HTML_DATA=soup.find_all("tbody")[0].find_all("tr")[:]
    for element in HTML_DATA:
        sub_data=[]
        for sub_element in element:
            try:
                if sub_element.get_text() != "\n":
                    datalines=sub_element.get_text().replace("+","")
                    sub_data.append(datalines)
            except:
                continue
        data.append(sub_data)
    with io.open("output.csv","w",encoding="utf-8")as f1:
        f1.write("PlayerName,Duration_in_years,MatchPlayed,Number_of_Innings,NotOut_Count,TotalRuns,HighestScore,Average,Number_of_Balls_Fcaed,Strike_Rate,1OO's_count,50's Count,zero count,fours count,sixers count"+"\n")
        for rows in data:
            row=",".join(rows)
            f1.write(row+"\n")
        f1.close()

except Exception as e:
    print(str(e))
                                                 
                                                 
