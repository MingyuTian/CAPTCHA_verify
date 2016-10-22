# CAPTCHA_verify
verify CAPTCHA of the website http://gsxt.gdgs.gov.cn/
There are two cases: the first one is calculation case. It's like '1+9=?' but in Chinese characters. The second case is over 100 fixed four-character word. Based on KNN method, I can predict more than 98% of the CAPTCHA correctly.

guangdong_verify.py is the code for it and *.npys are training data files.
