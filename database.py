"""
Database module for storing search history with multi-model support
Supports both SQLite (local) and PostgreSQL (Render)
"""

import os
from datetime import datetime, timedelta
from contextlib import contextmanager

# Check if PostgreSQL URL is available (Render environment)
DATABASE_URL = os.environ.get('DATABASE_URL')

# Determine database type
if DATABASE_URL:
    # PostgreSQL on Render
    USE_POSTGRES = True
    # Fix for psycopg2: Render uses 'postgres://' but psycopg2 needs 'postgresql://'
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    print("📦 Using PostgreSQL database")
else:
    # SQLite for local development
    USE_POSTGRES = False
    DATABASE_PATH = 'search_history.db'
    print("📦 Using SQLite database")

# Import appropriate database library
if USE_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor
else:
    import sqlite3


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    if USE_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL)
        try:
            yield conn
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()


def get_cursor(conn):
    """Get appropriate cursor based on database type"""
    if USE_POSTGRES:
        return conn.cursor(cursor_factory=RealDictCursor)
    else:
        return conn.cursor()


def get_row_value(row, key):
    """Get value from row (works for both dict and sqlite3.Row)"""
    if isinstance(row, dict):
        return row.get(key)
    else:
        return row[key]


def init_db():
    """Initialize the database with required tables"""
    with get_db_connection() as conn:
        cursor = get_cursor(conn)
        
        if USE_POSTGRES:
            # PostgreSQL syntax
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS search_history (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    airline TEXT,
                    origin TEXT,
                    destination TEXT,
                    flight_date TEXT,
                    return_date TEXT,
                    class TEXT,
                    trip_type TEXT,
                    predicted_price REAL,
                    model_type TEXT DEFAULT 'ann',
                    wifi TEXT,
                    meals TEXT,
                    baggage_kg INTEGER,
                    user_ip TEXT,
                    user_agent TEXT
                )
            ''')
            conn.commit()
            print("   ✅ PostgreSQL table ready")
        else:
            # SQLite syntax
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='search_history'")
            table_exists = cursor.fetchone()
            
            if not table_exists:
                cursor.execute('''
                    CREATE TABLE search_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        airline TEXT,
                        origin TEXT,
                        destination TEXT,
                        flight_date TEXT,
                        return_date TEXT,
                        class TEXT,
                        trip_type TEXT,
                        predicted_price REAL,
                        model_type TEXT DEFAULT 'ann',
                        wifi TEXT,
                        meals TEXT,
                        baggage_kg INTEGER,
                        user_ip TEXT,
                        user_agent TEXT
                    )
                ''')
                print("   ✅ SQLite table created")
            else:
                # Add missing columns for existing SQLite database
                cursor.execute('PRAGMA table_info(search_history)')
                existing_columns = [row[1] for row in cursor.fetchall()]
                
                columns_to_add = [
                    ('timestamp', 'DATETIME DEFAULT CURRENT_TIMESTAMP'),
                    ('model_type', 'TEXT DEFAULT "ann"'),
                    ('wifi', 'TEXT'),
                    ('meals', 'TEXT'),
                    ('baggage_kg', 'INTEGER'),
                    ('user_ip', 'TEXT'),
                    ('user_agent', 'TEXT'),
                    ('return_date', 'TEXT'),
                    ('trip_type', 'TEXT')
                ]
                
                for col_name, col_type in columns_to_add:
                    if col_name not in existing_columns:
                        try:
                            cursor.execute(f'ALTER TABLE search_history ADD COLUMN {col_name} {col_type}')
                            print(f"   ✅ Added column: {col_name}")
                        except Exception as e:
                            print(f"   ⚠️ Could not add column {col_name}: {e}")
            
            conn.commit()


# Initialize database on module import
try:
    init_db()
except Exception as e:
    print(f"   ⚠️ Database initialization warning: {e}")


def add_search(data, predicted_price, user_ip=None, user_agent=None, wifi=None, meals=None, baggage_kg=None, model_type='ann'):
    """Add a new search record"""
    try:
        with get_db_connection() as conn:
            cursor = get_cursor(conn)
            
            if USE_POSTGRES:
                cursor.execute('''
                    INSERT INTO search_history 
                    (airline, origin, destination, flight_date, return_date, class, trip_type, 
                     predicted_price, model_type, wifi, meals, baggage_kg, user_ip, user_agent, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    RETURNING id
                ''', (
                    data.get('airline'),
                    data.get('origin'),
                    data.get('destination'),
                    data.get('flight_date'),
                    data.get('return_date'),
                    data.get('class'),
                    data.get('trip_type', 'oneway'),
                    predicted_price,
                    model_type,
                    wifi,
                    meals,
                    baggage_kg,
                    user_ip,
                    user_agent
                ))
                result = cursor.fetchone()
                last_id = result['id'] if result else None
            else:
                cursor.execute('''
                    INSERT INTO search_history 
                    (airline, origin, destination, flight_date, return_date, class, trip_type, 
                     predicted_price, model_type, wifi, meals, baggage_kg, user_ip, user_agent, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ''', (
                    data.get('airline'),
                    data.get('origin'),
                    data.get('destination'),
                    data.get('flight_date'),
                    data.get('return_date'),
                    data.get('class'),
                    data.get('trip_type', 'oneway'),
                    predicted_price,
                    model_type,
                    wifi,
                    meals,
                    baggage_kg,
                    user_ip,
                    user_agent
                ))
                last_id = cursor.lastrowid
            
            conn.commit()
            print(f"   ✅ Search saved to database (ID: {last_id}, Model: {model_type})")
            return last_id
    except Exception as e:
        print(f"   ❌ Error saving search: {e}")
        return None


def get_statistics(model_filter='all'):
    """Get comprehensive statistics, optionally filtered by model"""
    try:
        with get_db_connection() as conn:
            cursor = get_cursor(conn)
            
            # Placeholder syntax differs between PostgreSQL and SQLite
            ph = '%s' if USE_POSTGRES else '?'
            
            # Base WHERE clause for model filter
            model_condition = ""
            model_params = []
            if model_filter != 'all':
                model_condition = f"WHERE model_type = {ph}"
                model_params = [model_filter]
            
            # Total searches
            query = f'SELECT COUNT(*) as total FROM search_history {model_condition}'
            cursor.execute(query, model_params)
            row = cursor.fetchone()
            total_searches = get_row_value(row, 'total') if row else 0
            
            if total_searches == 0:
                return {
                    'total_searches': 0,
                    'avg_price': 0,
                    'routes': [],
                    'airlines': [],
                    'airline_prices': [],
                    'classes': {},
                    'timeline': [],
                    'recent_searches': [],
                    'model_distribution': {}
                }
            
            # Average price
            query = f'SELECT AVG(predicted_price) as avg FROM search_history {model_condition}'
            cursor.execute(query, model_params)
            row = cursor.fetchone()
            avg_price = get_row_value(row, 'avg') or 0
            
            # Top routes
            query = f'''
                SELECT origin || ' → ' || destination as route, COUNT(*) as count
                FROM search_history {model_condition}
                GROUP BY origin, destination
                ORDER BY count DESC
                LIMIT 10
            '''
            cursor.execute(query, model_params)
            routes = [{'route': get_row_value(row, 'route'), 'count': get_row_value(row, 'count')} for row in cursor.fetchall()]
            
            # Top airlines
            query = f'''
                SELECT airline, COUNT(*) as count
                FROM search_history {model_condition}
                GROUP BY airline
                ORDER BY count DESC
                LIMIT 10
            '''
            cursor.execute(query, model_params)
            airlines = [{'airline': get_row_value(row, 'airline'), 'count': get_row_value(row, 'count')} for row in cursor.fetchall()]
            
            # Average price by airline
            query = f'''
                SELECT airline, AVG(predicted_price) as avg_price, COUNT(*) as count
                FROM search_history {model_condition}
                GROUP BY airline
                ORDER BY avg_price DESC
                LIMIT 10
            '''
            cursor.execute(query, model_params)
            airline_prices = [{'airline': get_row_value(row, 'airline'), 'avg_price': get_row_value(row, 'avg_price'), 'count': get_row_value(row, 'count')} for row in cursor.fetchall()]
            
            # Class distribution
            query = f'''
                SELECT class, COUNT(*) as count
                FROM search_history {model_condition}
                GROUP BY class
            '''
            cursor.execute(query, model_params)
            classes = {get_row_value(row, 'class'): get_row_value(row, 'count') for row in cursor.fetchall()}
            
            # Timeline (last 30 days)
            timeline = []
            try:
                if USE_POSTGRES:
                    query = f'''
                        SELECT DATE(timestamp) as date, COUNT(*) as count
                        FROM search_history
                        {model_condition}
                        {'AND' if model_condition else 'WHERE'} timestamp IS NOT NULL AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '30 days'
                        GROUP BY DATE(timestamp)
                        ORDER BY date
                    '''
                else:
                    query = f'''
                        SELECT DATE(timestamp) as date, COUNT(*) as count
                        FROM search_history
                        {model_condition}
                        {'AND' if model_condition else 'WHERE'} timestamp IS NOT NULL AND timestamp >= datetime('now', '-30 days')
                        GROUP BY DATE(timestamp)
                        ORDER BY date
                    '''
                cursor.execute(query, model_params)
                timeline = [{'date': str(get_row_value(row, 'date')), 'count': get_row_value(row, 'count')} for row in cursor.fetchall()]
            except Exception as e:
                print(f"Timeline query error: {e}")
            
            # Recent searches
            recent_searches = []
            try:
                if USE_POSTGRES:
                    query = f'''
                        SELECT 
                            COALESCE(timestamp::text, '') as timestamp, 
                            airline, origin, destination, flight_date, class, predicted_price, 
                            COALESCE(model_type, 'ann') as model_type
                        FROM search_history {model_condition}
                        ORDER BY id DESC
                        LIMIT 20
                    '''
                else:
                    query = f'''
                        SELECT 
                            COALESCE(timestamp, '') as timestamp, 
                            airline, origin, destination, flight_date, class, predicted_price, 
                            COALESCE(model_type, 'ann') as model_type
                        FROM search_history {model_condition}
                        ORDER BY id DESC
                        LIMIT 20
                    '''
                cursor.execute(query, model_params)
                recent_searches = [{
                    'timestamp': get_row_value(row, 'timestamp') or '',
                    'airline': get_row_value(row, 'airline'),
                    'origin': get_row_value(row, 'origin'),
                    'destination': get_row_value(row, 'destination'),
                    'flight_date': get_row_value(row, 'flight_date'),
                    'class': get_row_value(row, 'class'),
                    'predicted_price': get_row_value(row, 'predicted_price'),
                    'model_type': get_row_value(row, 'model_type') or 'ann'
                } for row in cursor.fetchall()]
            except Exception as e:
                print(f"Recent searches query error: {e}")
            
            # Model distribution
            model_distribution = {'ann': 0, 'linear_regression': 0, 'decision_tree': 0}
            try:
                if USE_POSTGRES:
                    cursor.execute('''
                        SELECT COALESCE(model_type, 'ann') as model, COUNT(*) as count
                        FROM search_history
                        GROUP BY COALESCE(model_type, 'ann')
                    ''')
                else:
                    cursor.execute('''
                        SELECT COALESCE(model_type, 'ann') as model, COUNT(*) as count
                        FROM search_history
                        GROUP BY model_type
                    ''')
                for row in cursor.fetchall():
                    model = get_row_value(row, 'model') or 'ann'
                    model_distribution[model] = get_row_value(row, 'count')
            except Exception as e:
                print(f"Model distribution query error: {e}")
            
            return {
                'total_searches': total_searches,
                'avg_price': avg_price,
                'routes': routes,
                'airlines': airlines,
                'airline_prices': airline_prices,
                'classes': classes,
                'timeline': timeline,
                'recent_searches': recent_searches,
                'model_distribution': model_distribution
            }
    except Exception as e:
        print(f"Statistics error: {e}")
        return {
            'total_searches': 0,
            'avg_price': 0,
            'routes': [],
            'airlines': [],
            'airline_prices': [],
            'classes': {},
            'timeline': [],
            'recent_searches': [],
            'model_distribution': {}
        }


def get_model_comparison_stats():
    """Get statistics comparing different models"""
    try:
        with get_db_connection() as conn:
            cursor = get_cursor(conn)
            
            if USE_POSTGRES:
                cursor.execute('''
                    SELECT 
                        COALESCE(model_type, 'ann') as model,
                        COUNT(*) as total_searches,
                        AVG(predicted_price) as avg_price,
                        MIN(predicted_price) as min_price,
                        MAX(predicted_price) as max_price
                    FROM search_history
                    GROUP BY COALESCE(model_type, 'ann')
                ''')
            else:
                cursor.execute('''
                    SELECT 
                        COALESCE(model_type, 'ann') as model,
                        COUNT(*) as total_searches,
                        AVG(predicted_price) as avg_price,
                        MIN(predicted_price) as min_price,
                        MAX(predicted_price) as max_price
                    FROM search_history
                    GROUP BY model_type
                ''')
            
            return [{
                'model': get_row_value(row, 'model'),
                'total_searches': get_row_value(row, 'total_searches'),
                'avg_price': get_row_value(row, 'avg_price'),
                'min_price': get_row_value(row, 'min_price'),
                'max_price': get_row_value(row, 'max_price')
            } for row in cursor.fetchall()]
    except Exception as e:
        print(f"Model comparison error: {e}")
        return []


# Alias for backwards compatibility
db = type('db', (), {
    'add_search': staticmethod(add_search),
    'get_statistics': staticmethod(get_statistics),
    'get_model_comparison_stats': staticmethod(get_model_comparison_stats)
})()