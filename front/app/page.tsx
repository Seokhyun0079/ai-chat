"use client";
import { useState, KeyboardEvent } from "react";
import Image from "next/image";
import UserChatMessage from "./chat-message/user-chat-message";

interface Message {
  text: string;
  isUser: boolean;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    { text: "안녕하세요! 무엇을 도와드릴까요?", isUser: false },
  ]);
  const [inputText, setInputText] = useState("");

  const handleSendMessage = async () => {
    if (inputText.trim()) {
      // Send the message to the server
      try {
        const response = await fetch("/api/chat/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ prompt: inputText }),
        });

        console.log(response);

        if (!response.ok) {
          throw new Error("Network response was not ok");
        }

        // Optionally handle the response from the server
        const data = await response.json();
        console.log("Server response:", data);

        // Update the local state
        setMessages([
          ...messages,
          { text: inputText, isUser: true },
          { text: data.response, isUser: false },
        ]);
        setInputText("");
      } catch (error) {
        console.error("Error sending message:", error);
      }
    }
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100 dark:bg-gray-900">
      {/* Chat header */}
      <header className="p-4 bg-white dark:bg-gray-800 shadow">
        <h1 className="text-xl font-bold">AI Chat</h1>
      </header>

      {/* Chat messages area */}
      <main className="flex-1 p-4 overflow-y-auto">
        <div className="space-y-4">
          {messages.map((message, index) =>
            message.isUser ? (
              <UserChatMessage key={index} message={message} />
            ) : (
              <div key={index} className="flex items-start gap-2.5">
                <div className="flex flex-col gap-1 w-full max-w-[320px] leading-1.5 p-4 border-gray-200 bg-gray-100 rounded-e-xl rounded-es-xl dark:bg-gray-700">
                  <p className="text-sm font-normal text-gray-900 dark:text-white">
                    {message.text}
                  </p>
                </div>
              </div>
            )
          )}
        </div>
      </main>

      {/* Chat input area */}
      <footer className="p-4 bg-white dark:bg-gray-800">
        <div className="flex gap-2">
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyPress={handleKeyPress}
            className="flex-1 p-2 rounded-full border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
            placeholder="메시지를 입력하세요..."
          />
          <button
            onClick={handleSendMessage}
            className="px-4 py-2 bg-blue-500 text-white rounded-full hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            전송
          </button>
        </div>
      </footer>
    </div>
  );
}
