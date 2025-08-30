# Portfolio Creation Fix

## Issue Found
The portfolio creation was failing with a 500 error because:
- Frontend was sending `initialCapital` 
- Backend expects `initial_cash`

## Fix Applied
Updated the portfolio service to correctly map the field names:
```javascript
// Before
body: JSON.stringify(data)

// After  
body: JSON.stringify({
  name: data.name,
  initial_cash: data.initialCapital // Correct field name
})
```

## How to Create a Portfolio Now

1. **Click the 🔄 refresh button** in the Portfolio Summary card header
   - This will reload any existing portfolios

2. **If still no portfolios**, try creating a new one:
   - Click "Create Portfolio" button
   - Enter a name (e.g., "My Trading Portfolio")
   - Enter initial capital (e.g., 100000)
   - Click "Create Portfolio"

3. **After creation**:
   - You should see "Portfolio created successfully!" toast
   - Click the 🔄 refresh button to load the new portfolio
   - Or use the "Clear Cache & Refresh" button in the top-right

## Troubleshooting

If portfolio creation still fails:

1. **Check browser console** (F12) for errors:
   - Look for "Portfolio creation error:" messages
   - This will show the exact error from the backend

2. **Try clearing all data**:
   - Click "Clear Cache & Refresh" button
   - This forces fresh API calls

3. **Verify backend is working**:
   - The backend logs show portfolio queries are working
   - The issue was just the field name mismatch

## Technical Details

The backend expects this structure:
```json
{
  "name": "Portfolio Name",
  "initial_cash": 100000.0,
  "portfolio_type": "trading" // optional
}
```

The portfolio will be created with:
- Initial cash = Buying power = Total value
- Daily returns = 0
- Is active = true
- Is default = true (if first portfolio)

Your portfolio should now be created and visible!
