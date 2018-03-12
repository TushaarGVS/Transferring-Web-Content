from selenium import webdriver

driver = webdriver.Firefox()
driver.get("http://www.google.com")
elem = driver.find_element_by_xpath("//html")

# print all attributes of driver
dir(driver)

driver.close()
