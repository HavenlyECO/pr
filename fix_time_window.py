#!/usr/bin/env python3

def fix_main_py():
    """Fix the time window check in evaluate_h2h_all_tomorrow function"""
    
    with open('main.py', 'r') as f:
        content = f.read()
    
    # Find the problematic section and fix it
    old_code = """    if not testing_mode:
            today = datetime.utcnow()
            start_dt = datetime(today.year, today.month, today.day, 16, 0, 0)
            end_dt = start_dt + timedelta(hours=14)

        if not (start_dt <= commence_dt < end_dt):
            if verbose:
                print(f"  Skipped: commence_time {commence_dt} not in window {start_dt} to {end_dt}")
            continue"""
    
    new_code = """    # Always define time window variables
        today = datetime.utcnow()
        start_dt = datetime(today.year, today.month, today.day, 16, 0, 0)
        end_dt = start_dt + timedelta(hours=14)
        
        # Only apply window filter in non-testing mode
        if not testing_mode and not (start_dt <= commence_dt < end_dt):
            if verbose:
                print(f"  Skipped: commence_time {commence_dt} not in window {start_dt} to {end_dt}")
            continue"""
    
    # Replace the problematic code
    new_content = content.replace(old_code, new_code)
    
    # Save the updated file
    with open('main.py', 'w') as f:
        f.write(new_content)
    
    print("Fixed the time window check in main.py")

if __name__ == "__main__":
    fix_main_py()
