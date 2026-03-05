import time
import random
import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime, timedelta
import argparse
import subprocess
import platform
import warnings
import sys
import re
import os

warnings.filterwarnings("ignore")

OUTPUT_FILE = 'flight_price_data_new.csv'

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

CITY_NAMES = {
    "SGN": "TP HCM", "HAN": "HÀ NỘI", "DAD": "ĐÀ NẴNG", "PQC": "PHÚ QUỐC", 
    "CXR": "NHA TRANG", "VCA": "CẦN THƠ", "HPH": "HẢI PHÒNG",
    "BKK": "BANGKOK", "SIN": "SINGAPORE", "KUL": "KUALA LUMPUR", 
    "CGK": "JAKARTA", "MNL": "MANILA",
    "ICN": "SEOUL", "NRT": "TOKYO", "KIX": "OSAKA", "HND": "TOKYO",
    "TPE": "TAIPEI", "PEK": "BEIJING", "PVG": "SHANGHAI", "HKG": "HONG KONG",
    "SYD": "SYDNEY", "MEL": "MELBOURNE", "BNE": "BRISBANE",
    "DXB": "DUBAI", "DOH": "DOHA",
}

ROUTES = [
    # ĐÔNG NAM Á
    {"from": "SGN", "to": "BKK", "days_offset": 7, "seat_class": "ECONOMY"},
    {"from": "SGN", "to": "BKK", "days_offset": 7, "seat_class": "BUSINESS"},
    {"from": "HAN", "to": "BKK", "days_offset": 10, "seat_class": "ECONOMY"},
    {"from": "HAN", "to": "BKK", "days_offset": 10, "seat_class": "BUSINESS"},
    {"from": "BKK", "to": "SGN", "days_offset": 8, "seat_class": "ECONOMY"},
    {"from": "BKK", "to": "SGN", "days_offset": 8, "seat_class": "BUSINESS"},
    
    {"from": "SGN", "to": "SIN", "days_offset": 5, "seat_class": "ECONOMY"},
    {"from": "SGN", "to": "SIN", "days_offset": 5, "seat_class": "BUSINESS"},
    {"from": "HAN", "to": "SIN", "days_offset": 9, "seat_class": "ECONOMY"},
    {"from": "HAN", "to": "SIN", "days_offset": 9, "seat_class": "BUSINESS"},
    
    {"from": "SGN", "to": "KUL", "days_offset": 8, "seat_class": "ECONOMY"},
    {"from": "SGN", "to": "KUL", "days_offset": 8, "seat_class": "BUSINESS"},
    {"from": "HAN", "to": "KUL", "days_offset": 14, "seat_class": "ECONOMY"},
    {"from": "HAN", "to": "KUL", "days_offset": 14, "seat_class": "BUSINESS"},
    
    {"from": "SGN", "to": "CGK", "days_offset": 9, "seat_class": "ECONOMY"},
    {"from": "SGN", "to": "CGK", "days_offset": 9, "seat_class": "BUSINESS"},
    
    # ĐÔNG BẮC Á
    {"from": "SGN", "to": "ICN", "days_offset": 10, "seat_class": "ECONOMY"},
    {"from": "SGN", "to": "ICN", "days_offset": 10, "seat_class": "BUSINESS"},
    {"from": "HAN", "to": "ICN", "days_offset": 14, "seat_class": "ECONOMY"},
    {"from": "HAN", "to": "ICN", "days_offset": 14, "seat_class": "BUSINESS"},
    {"from": "ICN", "to": "SGN", "days_offset": 11, "seat_class": "ECONOMY"},
    {"from": "ICN", "to": "SGN", "days_offset": 11, "seat_class": "BUSINESS"},
    
    {"from": "SGN", "to": "NRT", "days_offset": 12, "seat_class": "ECONOMY"},
    {"from": "SGN", "to": "NRT", "days_offset": 12, "seat_class": "BUSINESS"},
    {"from": "HAN", "to": "NRT", "days_offset": 16, "seat_class": "ECONOMY"},
    {"from": "HAN", "to": "NRT", "days_offset": 16, "seat_class": "BUSINESS"},
    
    {"from": "SGN", "to": "TPE", "days_offset": 8, "seat_class": "ECONOMY"},
    {"from": "SGN", "to": "TPE", "days_offset": 8, "seat_class": "BUSINESS"},
    {"from": "HAN", "to": "TPE", "days_offset": 11, "seat_class": "ECONOMY"},
    {"from": "HAN", "to": "TPE", "days_offset": 11, "seat_class": "BUSINESS"},
    
    {"from": "SGN", "to": "HKG", "days_offset": 6, "seat_class": "ECONOMY"},
    {"from": "SGN", "to": "HKG", "days_offset": 6, "seat_class": "BUSINESS"},
    {"from": "HAN", "to": "HKG", "days_offset": 10, "seat_class": "ECONOMY"},
    {"from": "HAN", "to": "HKG", "days_offset": 10, "seat_class": "BUSINESS"},
    
    # NỘI ĐỊA
    {"from": "SGN", "to": "HAN", "days_offset": 4, "seat_class": "ECONOMY"},
    {"from": "SGN", "to": "HAN", "days_offset": 4, "seat_class": "BUSINESS"},
    {"from": "HAN", "to": "SGN", "days_offset": 5, "seat_class": "ECONOMY"},
    {"from": "HAN", "to": "SGN", "days_offset": 5, "seat_class": "BUSINESS"},
]

# Config - TỐI ƯU TỐC ĐỘ
CONFIG = {
    "MIN_WAIT_BETWEEN_ROUTES": 12,  # Giảm xuống 12s
    "MAX_WAIT_BETWEEN_ROUTES": 20,  # Giảm xuống 20s
    "PAGE_LOAD_TIMEOUT": 45,        # Giảm xuống 45s
    "SCROLL_TIMES": 3,              # Giảm xuống 3 lần
    "SCROLL_DELAY": 2,              # 2 giây
    "MAX_RETRY": 1,
    "BATCH_SIZE": 5,                # Tăng lên 5 routes/batch
    "SAVE_INTERVAL": 3,             # Lưu sau mỗi 3 routes
    "HEADLESS": False,
}

def save_to_csv(data_list, append=True):
    if not data_list:
        return
    
    try:
        df_new = pd.DataFrame(data_list)
        df_new = df_new[["Airline", "Origin", "Destination", "Day", "Month", "Year", 
                         "Weekday", "Departure_Hour", "Arrival_Hour", "Duration_Minutes", 
                         "Stops", "Class", "WiFi", "Meals", "Baggage_kg", "Price_VND"]]
        
        file_exists = os.path.isfile(OUTPUT_FILE)
        
        if append and file_exists:
            df_new.to_csv(OUTPUT_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
            print(f"   💾 APPEND {len(data_list)} chuyến")
        else:
            df_new.to_csv(OUTPUT_FILE, mode='w', header=True, index=False, encoding='utf-8-sig')
            print(f"   💾 TẠO MỚI {len(data_list)} chuyến")
        
        return True
    except Exception as e:
        print(f"   ❌ Lỗi lưu: {e}")
        return False

def get_traveloka_url(origin, dest, date_obj, seat_class="ECONOMY"):
    date_str = date_obj.strftime("%d-%m-%Y")
    url = f"https://www.traveloka.com/vi-vn/flight/fullsearch?ap={origin}.{dest}&dt={date_str}.NA&ps=1.0.0&sc={seat_class}"
    return url

def clean_airline_name(name):
    name_lower = name.lower()
    
    if "vietjet" in name_lower: return "Vietjet"
    if "vietnam airlines" in name_lower: return "Vietnam Airlines"
    if "bamboo" in name_lower: return "Bamboo Airways"
    if "vietravel" in name_lower: return "Vietravel Airlines"
    if "pacific airlines" in name_lower: return "Pacific Airlines"
    if "airasia" in name_lower: return "AirAsia"
    if "jetstar" in name_lower: return "Jetstar"
    if "scoot" in name_lower: return "Scoot"
    if "thai" in name_lower: return "Thai Airways"
    if "singapore airlines" in name_lower: return "Singapore Airlines"
    if "malaysia airlines" in name_lower: return "Malaysia Airlines"
    if "garuda" in name_lower: return "Garuda Indonesia"
    if "korean air" in name_lower: return "Korean Air"
    if "asiana" in name_lower: return "Asiana Airlines"
    if "jal" in name_lower or "japan airlines" in name_lower: return "Japan Airlines"
    if "ana" in name_lower: return "ANA"
    if "eva air" in name_lower: return "EVA Air"
    if "china airlines" in name_lower: return "China Airlines"
    if "cathay" in name_lower: return "Cathay Pacific"
    if "emirates" in name_lower: return "Emirates"
    if "qatar" in name_lower: return "Qatar Airways"
    if "etihad" in name_lower: return "Etihad Airways"
    if "united" in name_lower: return "United Airlines"
    
    return name

def extract_baggage_from_icon(element, seat_class="ECONOMY"):
    """
    Trích xuất baggage từ icon/text - QUAN TRỌNG
    Tìm pattern: 1x32kg, 2x23kg, etc.
    """
    try:
        # Tìm trong HTML
        html = element.get_attribute('outerHTML')
        
        # Pattern: 1x32kg, 2x23kg, 1x40kg
        baggage_patterns = [
            r'(\d+)x(\d+)kg',
            r'(\d+)\s*x\s*(\d+)\s*kg',
        ]
        
        for pattern in baggage_patterns:
            match = re.search(pattern, html, re.IGNORECASE)
            if match:
                num_bags = int(match.group(1))
                kg_per_bag = int(match.group(2))
                return kg_per_bag  # Return kg per bag
        
        # Tìm trong text
        text = element.text
        for pattern in baggage_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                kg_per_bag = int(match.group(2))
                return kg_per_bag
        
        # Fallback theo seat class
        if seat_class == "BUSINESS":
            return 32
        else:
            return 20
            
    except:
        return 32 if seat_class == "BUSINESS" else 20

def extract_amenities_from_icons(element, seat_class="ECONOMY"):
    """
    Parse amenities từ ICONS - HOÀN TOÀN MỚI
    """
    amenities = {
        'meals': 'No',
        'wifi': 'No',
        'baggage': extract_baggage_from_icon(element, seat_class)
    }
    
    try:
        # Lấy HTML để tìm icons
        html = element.get_attribute('outerHTML').lower()
        text = element.text.lower()
        
        # 1. MEAL - Tìm icon hoặc text
        meal_indicators = [
            'meal', 'food', 'suất ăn', 'bữa ăn', 
            '🍽', '🍴', '🍜', '🍱',  # Unicode emoji
            'restaurant', 'dining', 'snack'
        ]
        
        for indicator in meal_indicators:
            if indicator in html or indicator in text:
                # Kiểm tra xem có phải "no meal" không
                if 'no meal' not in text and 'không có' not in text:
                    amenities['meals'] = 'Yes'
                    break
        
        # 2. WIFI - Tìm icon hoặc text  
        wifi_indicators = [
            'wifi', 'wi-fi', 'internet', 'wireless',
            '📶', '📡', '💻', '📺',  # Unicode emoji
            'entertainment', 'inflight', 'video'
        ]
        
        for indicator in wifi_indicators:
            if indicator in html or indicator in text:
                if 'no wifi' not in text and 'không có' not in text:
                    amenities['wifi'] = 'Yes'
                    break
        
        # 3. Nếu là BUSINESS class, default có meal & wifi
        if seat_class == "BUSINESS":
            if amenities['meals'] == 'No':
                amenities['meals'] = 'Yes'  # Business thường có meal
            if amenities['wifi'] == 'No':
                amenities['wifi'] = 'Limited'  # Business thường có wifi
        
    except Exception as e:
        # Default safe values
        if seat_class == "BUSINESS":
            amenities['meals'] = 'Yes'
            amenities['wifi'] = 'Limited'
    
    return amenities

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def kill_chrome_processes():
    try:
        if platform.system() == "Windows":
            subprocess.run("taskkill /F /IM chrome.exe /T", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            subprocess.run("taskkill /F /IM chromedriver.exe /T", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        else:
            subprocess.run("pkill -9 chrome", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            subprocess.run("pkill -9 chromedriver", shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        time.sleep(1)
    except:
        pass

def create_driver(user_agent=None, headless=False):
    try:
        options = uc.ChromeOptions()
        
        if headless:
            options.add_argument('--headless=new')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
        
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-popup-blocking')
        options.add_argument('--disable-notifications')
        options.add_argument('--start-maximized')
        options.add_argument('--disable-gpu')
        
        prefs = {
            "profile.default_content_setting_values.notifications": 2,
            "profile.managed_default_content_settings.images": 2,  # Tắt hình ảnh = nhanh hơn
        }
        options.add_experimental_option("prefs", prefs)
        
        if user_agent:
            options.add_argument(f'--user-agent={user_agent}')
        
        driver = uc.Chrome(options=options, version_main=143)
        driver.set_page_load_timeout(CONFIG["PAGE_LOAD_TIMEOUT"])
        
        return driver
    except Exception as e:
        print(f"❌ Lỗi tạo driver: {e}")
        return None

def safe_quit_driver(driver):
    if not driver:
        return
    try:
        driver.quit()
    except:
        pass
    time.sleep(1)

def random_scroll(driver, times=3, delay=2):
    """Scroll nhanh"""
    try:
        for i in range(times):
            scroll_height = random.randint(500, 1000)
            driver.execute_script(f"window.scrollBy(0, {scroll_height});")
            time.sleep(delay)
        
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(delay)
    except:
        pass

def crawl_single_route(driver, route, retry_count=0):
    """Crawl một route - TỐI ƯU"""
    flight_data = []
    max_retries = CONFIG["MAX_RETRY"]
    
    try:
        origin = route['from']
        dest = route['to']
        seat_class = route.get('seat_class', 'ECONOMY')
        days_offset = route.get('days_offset', 7)
        
        flight_date = datetime.now() + timedelta(days=days_offset)
        url = get_traveloka_url(origin, dest, flight_date, seat_class)
        
        print(f"   🛫 {origin}→{dest} | {seat_class} | {flight_date.strftime('%d/%m')}")
        
        driver.get(url)
        time.sleep(random.uniform(6, 10))  # Giảm thời gian chờ
        
        # Scroll nhanh
        random_scroll(driver, times=CONFIG["SCROLL_TIMES"], delay=CONFIG["SCROLL_DELAY"])
        time.sleep(3)
        
        # Tìm elements - ƯU TIÊN SELECTOR TỐT
        price_elements = []
        selectors = [
            "div[data-testid*='flight']",
            "div[class*='FlightCard']",
            "div[class*='flight']",
        ]
        
        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements and len(elements) >= 5:
                    price_elements = elements
                    print(f"   ✅ Found {len(price_elements)} elements")
                    break
            except:
                continue
        
        # Fallback
        if not price_elements or len(price_elements) < 5:
            try:
                all_divs = driver.find_elements(By.TAG_NAME, "div")
                price_elements = [d for d in all_divs if 'VND' in d.text and 50 < len(d.text) < 800]
                print(f"   ✅ Fallback: {len(price_elements)} elements")
            except:
                pass
        
        if not price_elements or len(price_elements) < 3:
            print(f"   ⚠️ Chỉ có {len(price_elements)} flights")
            if retry_count < max_retries:
                time.sleep(10)
                return crawl_single_route(driver, route, retry_count + 1)
            return []
        
        # Parse data - XỬ LÝ NHIỀU HƠN
        success_count = 0
        seen_prices = set()
        
        for idx, element in enumerate(price_elements[:60], 1):  # Xử lý tối đa 60
            try:
                element_text = element.text
                
                if not element_text or len(element_text) < 20:
                    continue
                
                # 1. GIÁ
                price = None
                price_patterns = [
                    r'([\d\.]+)\s*VND',
                    r'(\d{1,3}(?:\.\d{3}){1,})',
                ]
                
                for pattern in price_patterns:
                    price_match = re.search(pattern, element_text)
                    if price_match:
                        price_str = price_match.group(1).replace('.', '').replace(',', '')
                        try:
                            price = float(price_str)
                            if 100000 <= price <= 500000000:
                                break
                            else:
                                price = None
                        except:
                            continue
                
                if not price or price < 100000:
                    continue
                
                # Tránh duplicate
                if price in seen_prices:
                    continue
                seen_prices.add(price)
                
                # 2. HÃNG BAY
                airline = "Unknown"
                airline_patterns = [
                    r'(VietJet|Vietnam Airlines|Bamboo|Vietravel|Pacific Airlines)',
                    r'(AirAsia|Jetstar|Thai Airways|Singapore Airlines|United)',
                    r'(Korean Air|ANA|JAL|Cathay|Qatar|Emirates|Scoot)',
                    r'(EVA Air|China Airlines|Malaysia Airlines|Asiana|Garuda)',
                ]
                
                for pattern in airline_patterns:
                    match = re.search(pattern, element_text, re.IGNORECASE)
                    if match:
                        airline = clean_airline_name(match.group(1))
                        break
                
                # 3. GIỜ BAY
                time_pattern = r'(\d{1,2}):(\d{2})'
                times = re.findall(time_pattern, element_text)
                dep_hour = int(times[0][0]) if len(times) >= 1 else 0
                arr_hour = int(times[1][0]) if len(times) >= 2 else 0
                
                # 4. THỜI GIAN BAY
                duration_minutes = 0
                duration_match = re.search(r'(\d+)h\s*(\d+)?m?', element_text)
                if duration_match:
                    hours = int(duration_match.group(1))
                    minutes = int(duration_match.group(2)) if duration_match.group(2) else 0
                    duration_minutes = hours * 60 + minutes
                
                # 5. ĐIỂM DỪNG
                stops_match = re.search(r'(\d+)\s*(stop|điểm dừng|transit)', element_text, re.IGNORECASE)
                stops = int(stops_match.group(1)) if stops_match else 0
                
                # 6. AMENITIES - DÙNG HÀM MỚI
                amenities = extract_amenities_from_icons(element, seat_class)
                
                # 7. NGÀY
                day = flight_date.day
                month = flight_date.month
                year = flight_date.year
                weekday = flight_date.weekday()
                
                # 8. TÊN THÀNH PHỐ
                origin_city = CITY_NAMES.get(origin, origin)
                dest_city = CITY_NAMES.get(dest, dest)
                
                flight_record = {
                    "Airline": airline,
                    "Origin": origin_city,
                    "Destination": dest_city,
                    "Day": float(day),
                    "Month": float(month),
                    "Year": float(year),
                    "Weekday": float(weekday),
                    "Departure_Hour": float(dep_hour),
                    "Arrival_Hour": float(arr_hour),
                    "Duration_Minutes": float(duration_minutes),
                    "Stops": stops,
                    "Class": seat_class,
                    "WiFi": amenities['wifi'],
                    "Meals": amenities['meals'],
                    "Baggage_kg": amenities['baggage'],
                    "Price_VND": float(price)
                }
                
                flight_data.append(flight_record)
                success_count += 1
                
            except Exception as e:
                continue
        
        print(f"   ✅ Lưu: {success_count} flights")
        return flight_data
        
    except Exception as e:
        print(f"   ❌ Lỗi: {str(e)[:60]}")
        return []

def crawl_traveloka():
    """Main crawler - TỐI ƯU TỐC ĐỘ"""
    print(">>> 🚀 Traveloka Crawler V4 FINAL (Fast + Fix WiFi/Meal)")
    print(f">>> 📊 Routes: {len(ROUTES)}")
    print(f">>> ⚡ Tối ưu: Nhanh hơn 2x, lưu nhiều hơn")
    print("="*60)
    
    all_data = []
    temp_batch_data = []
    driver = None
    route_counter = 0
    
    try:
        if os.path.isfile(OUTPUT_FILE):
            existing_df = pd.read_csv(OUTPUT_FILE)
            existing_count = len(existing_df)
            print(f">>> 📂 File có: {existing_count} records")
        else:
            existing_count = 0
            print(f">>> 📂 File mới")
    except:
        existing_count = 0
    
    print("="*60)
    
    start_time = time.time()
    
    try:
        for idx, route in enumerate(ROUTES, 1):
            print(f"\n📍 {idx}/{len(ROUTES)}")
            
            if driver is None:
                try:
                    current_ua = get_random_user_agent()
                    driver = create_driver(user_agent=current_ua, headless=CONFIG["HEADLESS"])
                    if driver:
                        driver.delete_all_cookies()
                except Exception as e:
                    print(f"   ❌ Driver: {e}")
                    time.sleep(5)
                    continue
            
            if not driver:
                continue
            
            try:
                flight_data = crawl_single_route(driver, route)
                
                if flight_data:
                    temp_batch_data.extend(flight_data)
                    all_data.extend(flight_data)
                    route_counter += 1
                    
                    # AUTO-SAVE
                    if route_counter >= CONFIG["SAVE_INTERVAL"]:
                        print(f"\n💾 AUTO-SAVE: {len(temp_batch_data)} flights")
                        save_to_csv(temp_batch_data, append=True)
                        temp_batch_data = []
                        route_counter = 0
                
                # Wait giữa routes - GIẢM THỜI GIAN
                if idx < len(ROUTES):
                    wait_time = random.uniform(CONFIG["MIN_WAIT_BETWEEN_ROUTES"], 
                                               CONFIG["MAX_WAIT_BETWEEN_ROUTES"])
                    print(f"⏳ {wait_time:.0f}s...")
                    time.sleep(wait_time)
                
                # Restart driver
                if idx % CONFIG["BATCH_SIZE"] == 0:
                    print("\n🔄 Restart...")
                    safe_quit_driver(driver)
                    driver = None
                    kill_chrome_processes()
                    time.sleep(3)
                    
            except Exception as e:
                print(f"   ❌ {str(e)[:60]}")
                safe_quit_driver(driver)
                driver = None
                time.sleep(3)
                continue

    except KeyboardInterrupt:
        print("\n\n🛑 Dừng!")
        
    except Exception as e:
        print(f"\n❌ Lỗi: {e}")
        
    finally:
        if temp_batch_data:
            print(f"\n💾 FINAL: {len(temp_batch_data)} flights")
            save_to_csv(temp_batch_data, append=True)
        
        if driver:
            safe_quit_driver(driver)
        
        kill_chrome_processes()
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*60)
    print(f">>> ✅ DONE!")
    print(f">>> ⏱️  Thời gian: {elapsed/60:.1f} phút")
    print(f">>> 📊 Crawled: {len(all_data)} flights")
    
    try:
        df_final = pd.read_csv(OUTPUT_FILE)
        total = len(df_final)
        print(f">>> 📂 File: {total} records (was {existing_count})")
        print(f">>> ➕ Thêm: {total - existing_count} records")
        
        print(f"\n>>> 💺 Class:")
        print(df_final['Class'].value_counts())
        
        print(f"\n>>> 🧳 Baggage:")
        print(df_final.groupby('Class')['Baggage_kg'].value_counts().head(10))
        
        print(f"\n>>> 🍽️ Meals:")
        print(df_final['Meals'].value_counts())
        
        print(f"\n>>> 📶 WiFi:")
        print(df_final['WiFi'].value_counts())
        
    except Exception as e:
        print(f">>> ⚠️ {e}")
    
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--loop', action='store_true')
    parser.add_argument('--times', type=int, default=None)
    parser.add_argument('--wait', type=int, default=600)
    parser.add_argument('--headless', action='store_true')
    
    args = parser.parse_args()
    
    if args.headless:
        CONFIG["HEADLESS"] = True
    
    if args.loop:
        print("\n🔁 LOOP MODE")
        
        loop_count = 0
        
        try:
            while True:
                loop_count += 1
                print(f"\n{'='*60}")
                print(f"🔄 LOOP {loop_count}" + (f"/{args.times}" if args.times else ""))
                print(f"{'='*60}\n")
                
                crawl_traveloka()
                
                if args.times and loop_count >= args.times:
                    break
                
                print(f"\n💤 Sleep {args.wait}s...")
                time.sleep(args.wait)
                
        except KeyboardInterrupt:
            print(f"\n🛑 Dừng ở loop {loop_count}")
    
    else:
        crawl_traveloka()