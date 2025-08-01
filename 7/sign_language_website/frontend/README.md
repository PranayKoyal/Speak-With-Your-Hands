# Sign Language Recognition Frontend

This is the frontend application for the Sign Language Recognition system. It provides a web interface for real-time sign language recognition using your webcam.

## Features

- Real-time sign language recognition through webcam
- Text-to-speech output for recognized gestures
- Sentence building from recognized gestures
- Adjustable recognition settings
- Modern, responsive UI built with React and TailwindCSS

## Prerequisites

- Node.js (v14 or higher)
- npm (v6 or higher)
- A modern web browser with webcam support
- Backend server running (see backend README)

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Create a `.env` file in the root directory and add:
   ```
   REACT_APP_API_URL=http://localhost:5000
   ```

3. Start the development server:
   ```bash
   npm start
   ```

The application will be available at `http://localhost:3000`.

## Usage

1. Navigate to the Recognition page
2. Allow webcam access when prompted
3. Click "Start Camera" to begin recognition
4. Perform sign language gestures in front of the camera
5. Watch as gestures are recognized and added to the sentence
6. Use the "Speak" button to hear the sentence spoken aloud
7. Adjust settings as needed in the Settings page

## Project Structure

```
src/
├── components/     # React components
├── context/       # React Context providers
├── hooks/         # Custom React hooks
├── pages/         # Page components
├── services/      # API and other services
└── utils/         # Utility functions
```

## Available Scripts

- `npm start` - Runs the app in development mode
- `npm test` - Launches the test runner
- `npm run build` - Builds the app for production
- `npm run eject` - Ejects from Create React App

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
