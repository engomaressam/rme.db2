import mysql.connector as mysql
import pandas as pd
import os

class DatabaseConnection:
    def __init__(self):
        # Connection details
        self.host = "10.10.11.242"
        self.user = "omar2"
        self.password = "Omar_54321"
        self.database = "RME_TEST"
        self.connection = None
        self.cursor = None

    def connect(self):
        """Establish database connection"""
        try:
            print("Connecting to database...")
            self.connection = mysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            self.cursor = self.connection.cursor()
            print("Connected to MySQL database")
            return True
        except Exception as e:
            print(f"Error connecting to database: {str(e)}")
            return False

    def disconnect(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("\nMySQL connection is closed")

    def get_cost_data(self, year=2025, month=None):
        """
        Fetch cost data from RME_Projects_Cost_Dist_Line_Report
        
        Parameters:
        - year: Year to filter on (default 2025)
        - month: Optional month number (1-12)
        
        Returns DataFrame with the results
        """
        try:
            if month:
                query = """
                SELECT *
                FROM RME_Projects_Cost_Dist_Line_Report
                WHERE YEAR(GL_DATE) = %s
                AND MONTH(GL_DATE) = %s
                """
                self.cursor.execute(query, (year, month))
            else:
                query = """
                SELECT *
                FROM RME_Projects_Cost_Dist_Line_Report
                WHERE YEAR(GL_DATE) = %s
                """
                self.cursor.execute(query, (year,))

            columns = [desc[0] for desc in self.cursor.description]
            data = pd.DataFrame(self.cursor.fetchall(), columns=columns)
            print(f"Fetched {len(data)} cost records")
            return data
        except Exception as e:
            print(f"Error fetching cost data: {str(e)}")
            return pd.DataFrame()

    def get_cash_data(self, year=2025, month=None):
        """
        Fetch cash data from RME_ap_check_payments_Report
        
        Parameters:
        - year: Year to filter on (default 2025)
        - month: Optional month number (1-12)
        
        Returns DataFrame with the results
        """
        try:
            if month:
                query = """
                SELECT *
                FROM RME_ap_check_payments_Report
                WHERE YEAR(CHECK_DATE) = %s
                AND MONTH(CHECK_DATE) = %s
                """
                self.cursor.execute(query, (year, month))
            else:
                query = """
                SELECT *
                FROM RME_ap_check_payments_Report
                WHERE YEAR(CHECK_DATE) = %s
                """
                self.cursor.execute(query, (year,))

            columns = [desc[0] for desc in self.cursor.description]
            data = pd.DataFrame(self.cursor.fetchall(), columns=columns)
            print(f"Fetched {len(data)} cash records")
            return data
        except Exception as e:
            print(f"Error fetching cash data: {str(e)}")
            return pd.DataFrame()

def main():
    # Example usage
    db = DatabaseConnection()
    if db.connect():
        try:
            # Example: Get January 2025 data
            cost_jan = db.get_cost_data(year=2025, month=1)
            cash_jan = db.get_cash_data(year=2025, month=1)
            
            print("\nCost data columns:", cost_jan.columns.tolist())
            print("Cash data columns:", cash_jan.columns.tolist())
            
        finally:
            db.disconnect()

if __name__ == "__main__":
    main() 