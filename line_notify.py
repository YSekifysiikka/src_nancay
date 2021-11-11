#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:17:37 2021

@author: yuichiro
"""


import requests

def main():
    send_line_notify('てすとてすと')

def send_line_notify(notification_message):
    """
    LINEに通知する
    """
    line_notify_token = 'er3U6S4bzTPrtjPtgHfYF0U4RBE4S8Ub9pWGxd511kC'
    line_notify_api = 'https://notify-api.line.me/api/notify'
    headers = {'Authorization': f'Bearer {line_notify_token}'}
    data = {'message': f'message: {notification_message}'}
    requests.post(line_notify_api, headers = headers, data = data)

if __name__ == "__main__":
    main()
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 19:04:34 2020

@author: yuichiro
"""
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from time import sleep
import time

print ('外資就活')
def scrapingLogin():
    options = Options()
    
    # Google Chrome Canaryのインストールパスを指定する
#    options.binary_location = '/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary'
    

    # Headless Chromeを使うためのオプション
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    driver = webdriver.Chrome('/Users/yuichiro/opt/anaconda3/lib/python3.7/site-packages/chromedriver_binary/chromedriver', chrome_options=options)
    
    # 設定したオプションを使ってwebdriverオブジェクトを作成
#    driver = webdriver.Chrome(chrome_options=options)

    # Google Chrome Canaryを起動してTwtitterのトップページに接続
    driver.get("https://gaishishukatsu.com/login")

    # あなたのユーザー名/メールアドレス
    username = 'yuseki219@gmail.com'
    # あなたのパスワード
    password = 'gfa1192393'

    # ユーザー名の入力ボックスを探す
    username_box = driver.find_element_by_xpath("//*[@id='GsUserEmail']")
    # パスワードの入力ボックスを探す
    password_box = driver.find_element_by_xpath("//*[@id='GsUserPassword']")

    # ユーザ名とパスワードをインプットする
    username_box.send_keys(username)
    password_box.send_keys(password)

    # ログインボタンを探す
    login_button = driver.find_element_by_xpath("//*[@id='GsUserLoginForm']/div[4]/p[1]/button")
    #ログインボタンをクリック
    login_button.click()

    # プログラムが動いたと判断するための待機時間
    sleep(3)
    
    driver.get('https://gaishishukatsu.com/recruiting_info?order=deadline&tagIDs=&typeIDs=&industryCategoryIDs=&gy=2022')
    time.sleep(3)

#    company_name = driver.find_element_by_class_name('sc-gpHHfC.kWUwDK').text
#    print (company_name)

    def ranking(driver):
        class_name = 'sc-gpHHfC.kWUwDK'                   # class属性名
        class_elems = driver.find_elements_by_class_name(class_name) # classでの指定
     
        # 取得した要素を1つずつ表示
        for elem in class_elems:
            print('企業名:-' + elem.text + '-')
     
    ranking(driver)
    # ブラウザを閉じる
    driver.close()
    # Google Chrome Canaryを終了する
    driver.quit()

    return "-------------------------------"

def main():
    # scrapingLogin関数を実行
    output = scrapingLogin()
    print(output)

# プログラム実行のおまじない
if __name__ == '__main__':
    main()



print ('one_career')



from bs4 import BeautifulSoup
import urllib.request as req
import time
url = 'https://www.onecareer.jp/events/internship?category=internship&order=&q%5Bcompany_name_like%5D=&q%5Bgrade_eq_or_null%5D=2022%E5%B9%B4%E5%8D%92&q%5Bschedules_prefecture_equal%5D=&schedule_target=start_at&schedules_begin_at=&schedules_end_at='
res = req.urlopen(url)
soup = BeautifulSoup(res, "html.parser")
company_list = []
i = 1
while True:
    if not i == 1:
        href_list = []
        text_list = []
        pages_list = soup.select('div.v2-events-pagination')
        for pages in pages_list:
            for a in pages.select("a"):
                href= a.attrs['href']
                text = a.string
                href_list.append(href)
                text_list.append(text)
        if url == 'https://www.onecareer.jp' + href_list[-1]:
            print ('Done')
            break
        else:
            url = 'https://www.onecareer.jp' + href_list[-1]
            res = req.urlopen(url)
            soup = BeautifulSoup(res, "html.parser")
    companies = soup.select('div.v2-featured-event__content')
    for company in companies:
        company_data = company.select('div.hidden-xs')
        for company_datum in company_data:
            company_name = company_datum.select('span.v2-event-company-list__item-company-name')
            if len(company_name) > 0:
                print ('企業名:-' + company_name[0].string + '-')
                company_list.append(company_name[0].string)
    
#        company_events = company.select('div.v2-event-schedule')
#        i = 0
#        for company_event in company_events:
#            i += 1
#            event_title = company_event.select('p.v2-event-schedule__title')
#            if len(event_title) > 0:
#                print (str(i) + 'イベント名:' + event_title[0].string)
#            event_term = company_event.select('span.v2-event-schedule__header-start-at-and-end-at')
#            if len(event_term) > 0:
#                print ('期間:' + event_term[0].string)
#            event_place = company_event.select('span.v2-event-schedule__header-prefecture')
#            if len(event_place) > 0:
#                print ('場所:' + event_place[0].string)
#            deadline = company_event.select('p.v2-event-schedule__time-limit-at > strong')
#    #        print (deadline)
#            if len(deadline) > 0:
#                print ('締め切り:' + deadline[0].string.rstrip('\n').lstrip('\n'))
#        print ('------------------------------')
    
    companies = soup.select('div.v2-pickup-event-list__item')
    #span_list = soup.select("div.v2-featured-event__info-top-left > div > div > ul > li > span")
    for company in companies:
        company_name = company.select('span.v2-event-company-list__item-company-name')
        if len(company_name) > 0:
            print ('企業名:-' + company_name[0].string + '-')
            company_list.append(company_name[0].string)
    companies = soup.select('div.v2-event-list__item')
    for company in companies:
        company_name = company.select('span.v2-event-company-list__item-company-name')
        if len(company_name) > 0:
            print ('企業名:-' + company_name[0].string + '-')
            company_list.append(company_name[0].string)
    i += 1
    time.sleep(2)