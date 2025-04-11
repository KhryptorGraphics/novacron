import { useState, useEffect } from "react";
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";

// OS Image interface
export interface OSImage {
  id: string;
  name: string;
  os: string;
  version: string;
  architecture: string;
  description?: string;
  path: string;
}

// Props interface
interface OSSelectorProps {
  value: string;
  onChange: (value: string) => void;
  className?: string;
}

// OS Selector component
export function OSSelector({ value, onChange, className }: OSSelectorProps) {
  const [images, setImages] = useState<OSImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // In a real application, this would fetch from the API
    // For now, we'll use mock data
    const fetchImages = async () => {
      setLoading(true);
      try {
        // Simulate API call
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Mock data
        const mockImages: OSImage[] = [
          {
            id: "ubuntu-20.04",
            name: "Ubuntu 20.04 LTS",
            os: "Ubuntu",
            version: "20.04",
            architecture: "x86_64",
            description: "Ubuntu 20.04 LTS (Focal Fossa)",
            path: "/var/lib/novacron/images/ubuntu-20.04-server-cloudimg-amd64.qcow2"
          },
          {
            id: "ubuntu-22.04",
            name: "Ubuntu 22.04 LTS",
            os: "Ubuntu",
            version: "22.04",
            architecture: "x86_64",
            description: "Ubuntu 22.04 LTS (Jammy Jellyfish)",
            path: "/var/lib/novacron/images/ubuntu-22.04-server-cloudimg-amd64.qcow2"
          },
          {
            id: "ubuntu-24.04",
            name: "Ubuntu 24.04 LTS",
            os: "Ubuntu",
            version: "24.04",
            architecture: "x86_64",
            description: "Ubuntu 24.04 LTS (Noble Numbat)",
            path: "/var/lib/novacron/images/ubuntu-24.04-server-cloudimg-amd64.qcow2"
          },
          {
            id: "centos-8",
            name: "CentOS 8",
            os: "CentOS",
            version: "8",
            architecture: "x86_64",
            description: "CentOS 8 Stream",
            path: "/var/lib/novacron/images/centos-8-x86_64.qcow2"
          },
          {
            id: "debian-11",
            name: "Debian 11",
            os: "Debian",
            version: "11",
            architecture: "x86_64",
            description: "Debian 11 (Bullseye)",
            path: "/var/lib/novacron/images/debian-11-generic-amd64.qcow2"
          }
        ];
        
        setImages(mockImages);
      } catch (err) {
        console.error("Failed to fetch OS images:", err);
        setError("Failed to load OS images. Please try again.");
      } finally {
        setLoading(false);
      }
    };
    
    fetchImages();
  }, []);

  return (
    <div className={className}>
      <Label htmlFor="os-selector" className="text-right">
        Operating System
      </Label>
      <Select 
        value={value} 
        onValueChange={onChange}
        disabled={loading}
      >
        <SelectTrigger id="os-selector" className="w-full">
          <SelectValue placeholder={loading ? "Loading..." : "Select an operating system"} />
        </SelectTrigger>
        <SelectContent>
          {error ? (
            <div className="p-2 text-red-500 text-sm">{error}</div>
          ) : loading ? (
            <div className="p-2 text-gray-500 text-sm">Loading...</div>
          ) : (
            <>
              <div className="p-2 text-xs font-semibold text-gray-500">Ubuntu</div>
              {images
                .filter(img => img.os === "Ubuntu")
                .map(img => (
                  <SelectItem key={img.id} value={img.id}>
                    <div className="flex flex-col">
                      <span>{img.name}</span>
                      <span className="text-xs text-gray-500">{img.description}</span>
                    </div>
                  </SelectItem>
                ))
              }
              
              <div className="p-2 text-xs font-semibold text-gray-500 mt-2">CentOS</div>
              {images
                .filter(img => img.os === "CentOS")
                .map(img => (
                  <SelectItem key={img.id} value={img.id}>
                    <div className="flex flex-col">
                      <span>{img.name}</span>
                      <span className="text-xs text-gray-500">{img.description}</span>
                    </div>
                  </SelectItem>
                ))
              }
              
              <div className="p-2 text-xs font-semibold text-gray-500 mt-2">Debian</div>
              {images
                .filter(img => img.os === "Debian")
                .map(img => (
                  <SelectItem key={img.id} value={img.id}>
                    <div className="flex flex-col">
                      <span>{img.name}</span>
                      <span className="text-xs text-gray-500">{img.description}</span>
                    </div>
                  </SelectItem>
                ))
              }
            </>
          )}
        </SelectContent>
      </Select>
    </div>
  );
}
