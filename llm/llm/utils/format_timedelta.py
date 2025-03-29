from datetime import timedelta


def format_time_delta(delta: timedelta) -> str:

    total_seconds = int(delta.total_seconds())
    days, remainder = divmod(total_seconds, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days == 0:
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    return f"{days} days, {hours:02}:{minutes:02}:{seconds:02}"
