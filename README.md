# Library Recommendation System

## Overview
The Library Recommendation System is a web application designed to help users discover books based on their preferences and interactions. The application allows users to register, log in, browse available books, and receive personalized book recommendations.

## Project Structure
```
library_recommendation_system
├── src
│   ├── app.py                  # Main entry point of the application
│   ├── pages                   # Contains different pages of the application
│   │   ├── register.py         # User registration logic
│   │   ├── login.py            # User login functionality
│   │   ├── recommendations.py   # Book recommendations display
│   │   └── browse.py           # Book browsing functionality
│   ├── components              # Reusable components for the application
│   │   ├── user_info.py        # User information display in the sidebar
│   │   ├── book_card.py        # Individual book card layout and functionality
│   │   ├── sidebar.py          # Sidebar layout and navigation
│   │   └── pagination.py       # Pagination logic for book display
│   ├── utils                   # Utility functions
│   │   ├── model.py            # Model loading and management
│   │   └── data.py             # Data loading and saving functions
│   └── types                   # Custom types and interfaces
│       └── index.py            # Type definitions
├── data                        # Data files
│   ├── users.json              # User data in JSON format
│   ├── books_with_intro_cleaned.csv # Book information in CSV format
│   ├── interactions.csv         # User interactions log
│   └── default_cover.jpeg       # Default cover image
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd library_recommendation_system
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run src/app.py
   ```

## Usage
- **Registration**: New users can register by providing their user ID, gender, age, and preferred genres.
- **Login**: Existing users can log in using their user ID.
- **Browse Books**: Users can browse through a list of available books, view details, and log interactions (e.g., view, favorite, reserve, borrow).
- **Recommendations**: Based on user preferences and interactions, personalized book recommendations are displayed.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License.