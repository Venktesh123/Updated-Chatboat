// Dhamm AI Chatbot API Test Script
// This script demonstrates how to properly use the Dhamm AI Chatbot API from a browser

// Test the chat API
async function testChatAPI() {
    try {
        const response = await fetch('http://localhost:5000/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                question: 'What is civil engineering?'
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('API Response:', data);
        
        // Display the answer in a more readable format
        console.log('\nAnswer from AI:\n', data.answer);
        console.log('\nCognitive Level:', data.cognitive_level, '-', data.cognitive_description);
        console.log('Sentiment:', data.sentiment);
        
        return data;
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
}

// Clear conversation history
async function clearConversation() {
    try {
        const response = await fetch('http://localhost:5000/api/clear', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Conversation cleared:', data);
        return data;
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
}

// Toggle showing source chunks
async function toggleChunks(showChunks) {
    try {
        const response = await fetch('http://localhost:5000/api/toggle-chunks', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify({
                show_chunks: showChunks
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Toggle chunks:', data);
        return data;
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
}

// Health check
async function healthCheck() {
    try {
        const response = await fetch('http://localhost:5000/', {
            method: 'GET'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Health check:', data);
        return data;
    } catch (error) {
        console.error('Error:', error);
        return null;
    }
}

// Run tests
console.log('Starting API tests...');

// Uncomment to run different tests
// healthCheck();
// testChatAPI();
// clearConversation();
// toggleChunks(true);

// Example of a conversation flow
async function runConversationDemo() {
    console.log('Running conversation demo...');
    
    // First clear any existing conversation
    await clearConversation();
    
    // Show source chunks for educational purposes
    await toggleChunks(true);
    
    // First question
    const response1 = await testChatAPI();
    
    // Follow-up question that builds on the first
    if (response1) {
        setTimeout(async () => {
            const response2 = await fetch('http://localhost:5000/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({
                    question: 'Can you explain more about the different branches of civil engineering?'
                })
            });
            
            if (response2.ok) {
                const data2 = await response2.json();
                console.log('\nFollow-up Answer:\n', data2.answer);
                console.log('\nCognitive Level:', data2.cognitive_level, '-', data2.cognitive_description);
                console.log('Sentiment:', data2.sentiment);
                
                if (data2.chunks && data2.chunks.length > 0) {
                    console.log('\nSource Chunks:');
                    data2.chunks.forEach((chunk, i) => {
                        console.log(`\nChunk ${i+1}:\n${chunk.substring(0, 150)}...`);
                    });
                }
            }
        }, 1000);
    }
}

// Uncomment to run the conversation demo
// runConversationDemo();

console.log('\nTo use this script:');
console.log('1. Open your browser console');
console.log('2. Uncomment the test function you want to run');
console.log('3. Or call the functions directly in the console');
console.log('\nExample: testChatAPI()');
