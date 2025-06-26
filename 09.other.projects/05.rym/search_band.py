from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import tkinter as tk
from tkinter import messagebox
import time

def get_band_name():
    # Create the main window
    root = tk.Tk()
    root.title('RateYourMusic Search')
    
    # Center the window
    window_width = 300
    window_height = 150
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f'{window_width}x{window_height}+{x}+{y}')
    
    # Variable to store the input
    band_var = tk.StringVar()
    
    # Create and pack widgets
    tk.Label(root, text='Enter band name:', pady=10).pack()
    entry = tk.Entry(root, textvariable=band_var, width=30)
    entry.pack(pady=5)
    
    # Function to handle submit
    def on_submit():
        root.quit()
    
    tk.Button(root, text='Search', command=on_submit).pack(pady=10)
    
    # Focus on entry and bind Enter key
    entry.focus()
    entry.bind('<Return>', lambda e: on_submit())
    
    # Run the window
    root.mainloop()
    
    # Get the value and destroy the window
    value = band_var.get()
    root.destroy()
    return value

def search_band(band_name):
    try:
        # Set up Chrome options
        options = webdriver.ChromeOptions()
        options.add_argument('--start-maximized')
        
        # Create Chrome driver
        driver = webdriver.Chrome(options=options)
        
        # Navigate to RateYourMusic
        driver.get('https://rateyourmusic.com/')
        
        # Wait for search icon and click it
        wait = WebDriverWait(driver, 10)
        search_icon = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'searchIcon')))
        search_icon.click()
        
        # Wait for search input and enter band name
        search_input = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'searchInput')))
        search_input.send_keys(band_name)
        search_input.send_keys(Keys.RETURN)
        
        # Wait for results to load
        time.sleep(2)
        
        # Take screenshot
        driver.save_screenshot(f"{band_name.replace(' ', '_')}_search_results.png")
        messagebox.showinfo('Success', 'Search completed successfully!')
        
        # Keep the browser open for viewing
        input('Press Enter to close the browser...')
        
    except Exception as e:
        messagebox.showerror('Error', f'An error occurred: {str(e)}')
    finally:
        try:
            driver.quit()
        except:
            pass

if __name__ == '__main__':
    band_name = get_band_name()
    if band_name.strip():
        search_band(band_name)
    else:
        messagebox.showwarning('Warning', 'Please enter a band name to search.')
