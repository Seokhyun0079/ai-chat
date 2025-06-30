interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  timestamp?: Date;
}

export function exportChatToFile(
  messages: ChatMessage[],
  aiName: string,
  userName: string
) {
  // 현재 날짜와 시간을 가져와서 파일명 생성
  const now = new Date();
  const dateStr = now.toISOString().slice(0, 10).replace(/-/g, "");
  const timeStr = now.toTimeString().slice(0, 8).replace(/:/g, "");
  const fileName = `Chat_${dateStr}_${timeStr}_${aiName}_${userName}.txt`;

  // 카카오톡 형식으로 메시지 변환
  const chatContent = messages
    .map((message) => {
      const timestamp = message.timestamp || new Date();
      const timeStr = timestamp.toLocaleTimeString("ko-KR", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: true,
      });

      const senderName = message.role === "assistant" ? aiName : userName;
      return `[${senderName}] [${timeStr}] ${message.content}`;
    })
    .join("\n");

  // 파일 생성 및 다운로드
  const blob = new Blob([chatContent], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);

  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  URL.revokeObjectURL(url);
} 