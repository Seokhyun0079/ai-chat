"use client";
import { useState, useRef, useEffect, KeyboardEvent } from "react";
import Image from "next/image";
import UserChatMessage from "./chat-message/user-chat-message";

interface Message {
  text: string;
  isUser: boolean;
  id: string;
  visible: boolean;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    { text: "나다", isUser: false, id: "0", visible: true },
  ]);
  const [inputText, setInputText] = useState("");
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const addMessageWithDelay = (newMessages: Message[]) => {
    setMessages((prev) => [
      ...prev,
      ...newMessages.map((msg) => ({ ...msg, visible: false })),
    ]);

    // Show messages one by one with 0.5s delay
    newMessages.forEach((message, index) => {
      setTimeout(() => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === message.id ? { ...msg, visible: true } : msg
          )
        );
        scrollToBottom();
      }, index * 500);
    });
  };

  const handleSendMessage = async () => {
    if (inputText.trim()) {
      // Send the message to the server
      try {
        console.dir(messages[messages.length - 1].text);
        const response = await fetch("/api/chat/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            prompt: inputText,
            max_length: 30,
          }),
        });

        console.log(response);

        if (!response.ok) {
          throw new Error("Network response was not ok");
        }

        // Optionally handle the response from the server
        const data = await response.json();
        console.log("Server response:", data);

        // Split the response by newlines and create separate messages
        const split = data.response.split("\n");
        const responseLines = split.filter(
          (line: string, index: number) =>
            line.trim() !== "" && index !== split.length - 1
        );

        // Create new messages array with user message and bot responses
        const userMessage: Message = {
          text: inputText,
          isUser: true,
          id: Date.now().toString(),
          visible: false,
        };

        const botMessages: Message[] = responseLines.map(
          (line: string, index: number) => ({
            text: line,
            isUser: false,
            id: (Date.now() + index + 1).toString(),
            visible: false,
          })
        );

        // Add all messages with delay
        const allNewMessages = [userMessage, ...botMessages];
        addMessageWithDelay(allNewMessages);
        setInputText("");
      } catch (error) {
        console.error("Error sending message:", error);
      }
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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
              <div
                key={message.id}
                className={`transition-all duration-500 ease-in-out ${
                  message.visible
                    ? "opacity-100 transform translate-y-0"
                    : "opacity-0 transform translate-y-4"
                }`}
              >
                <UserChatMessage message={message} />
              </div>
            ) : (
              <div
                key={message.id}
                className={`transition-all duration-500 ease-in-out ${
                  message.visible
                    ? "opacity-100 transform translate-y-0"
                    : "opacity-0 transform translate-y-4"
                }`}
              >
                <div className="flex items-start gap-2.5">
                  <div className="flex flex-col gap-1 w-full max-w-[320px] leading-1.5 p-4 border-gray-200 bg-yellow-100 rounded-e-xl rounded-es-xl dark:bg-yellow-800">
                    <p className="text-sm font-normal text-gray-900 dark:text-white">
                      {message.text}
                    </p>
                  </div>
                </div>
              </div>
            )
          )}
          <div ref={messagesEndRef} />
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
