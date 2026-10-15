"use client";

import { useEffect, useMemo, useState } from "react";

import { cn } from "@/lib/utils";

type ImageThumbnailProps = {
  src: string;
  thumbnailSrc?: string;
  alt?: string;
  className?: string;
  imageClassName?: string;
};

export function upgradeToHttps(url: string) {
  if (typeof window === "undefined") return url;
  if (window.location.protocol !== "https:") return url;
  if (url.startsWith("https://")) return url;
  if (url.startsWith("http://")) return `https://${url.slice(7)}`;
  return url;
}

export function getImageThumbnailUrl(src: string) {
  const marker = "/images/";
  const index = src.indexOf(marker);
  if (index < 0) return upgradeToHttps(src);
  return upgradeToHttps(`${src.slice(0, index)}/image-thumbnails/${src.slice(index + marker.length)}`);
}

export function ImageThumbnail({ src, thumbnailSrc, alt = "", className, imageClassName }: ImageThumbnailProps) {
  const initialSrc = useMemo(() => upgradeToHttps(thumbnailSrc || getImageThumbnailUrl(src)), [src, thumbnailSrc]);
  const [currentSrc, setCurrentSrc] = useState(initialSrc);

  useEffect(() => {
    setCurrentSrc(initialSrc);
  }, [initialSrc]);

  return (
    <span className={cn("block overflow-hidden bg-stone-100", className)}>
      <img
        src={currentSrc}
        alt={alt}
        className={cn("h-full w-full object-cover", imageClassName)}
        loading="lazy"
        decoding="async"
        onError={() => {
          if (currentSrc !== src) {
            setCurrentSrc(upgradeToHttps(src));
          }
        }}
      />
    </span>
  );
}
