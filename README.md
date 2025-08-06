# Library Recommendation System

## Overview
The Library Recommendation System is a web application designed to help users discover books based on their preferences and interactions. The application allows users to register, log in, browse available books, and receive personalized book recommendations.

## Setup Instructions
1. __Clone the repository__
   ```
   git clone https://github.com/jennycs0830/library_recommendation_system.git
   cd library_recommendation_system
   ```

2. __Install the required dependencies__
   ```
   pip install -r requirements.txt
   ```

3. __Build docker images__
   - Autoencoder image: 
     ```
     docker build -f docker/Dockerfile.autoencoder -t autoencoder-api
     ```
   - Clustering image: 
      ```
      docker build -f docker/Dockerfile.clustering -t clustering-api .
      ```

4. __Run docker images__
   - Autoencoder API (http://localhost:8002/encode)
     ```
     docker run -d --name autoencoder-api -p 8002:8002 autoencoder-api
     ```
   - Clustering API (http://localhost:8003/encode)
     ```
     docker run -d --name clustering-api -p 8003:8003 clustering-api
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