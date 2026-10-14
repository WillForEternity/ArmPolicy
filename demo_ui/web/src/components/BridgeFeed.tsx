import { useState } from "react";

/** Annotated frames from the bridge (a runtime owns the camera). Asks for the next frame as soon as
 * one has loaded, so it runs as fast as the link allows without queueing stale frames. */
export default function BridgeFeed({ className, onSize }: { className?: string; onSize?: (w: number, h: number) => void }) {
  const [n, setN] = useState(0);
  return (
    <img
      className={className}
      src={`/frame.jpg?n=${n}`}
      alt=""
      onLoad={(e) => {
        onSize?.(e.currentTarget.naturalWidth, e.currentTarget.naturalHeight);
        requestAnimationFrame(() => setN((x) => x + 1));
      }}
      onError={() => setTimeout(() => setN((x) => x + 1), 500)}
    />
  );
}
