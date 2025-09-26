from datetime import datetime, timedelta
import requests


def datetime_from_str(date: str):
    return datetime.strptime(date, '%Y-%m-%d')

def datetime_to_str(date: datetime):
    return date.strftime('%Y-%m-%d')