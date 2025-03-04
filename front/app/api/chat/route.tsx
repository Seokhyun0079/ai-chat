import { NextResponse } from "next/server";

export async function GET(request: Request) {
  // ... 기존 코드 ...
  return NextResponse.json({ message: "Hello from GET" });
}

export async function POST(request: Request) {
  const { prompt, previous_message, max_length } = await request.json();
  console.log("prompt", prompt);
  console.log("previous_message", previous_message);
  console.log("max_length", max_length);
  const response = await fetch("http://localhost:8000/chat/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      prompt: prompt,
      previous_message: previous_message,
      max_length: max_length,
    }),
  });
  const data = await response.json();
  return NextResponse.json(data);
}
