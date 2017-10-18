% include header title=title
  <body>
    <div class="body">
        <h1>{{title}}</h1>
            <p>
              % for t in temperatures :
              {{t[0]}} : {{t[1]}} ({{t[2]}})<br>
              % end
            </p>
            <p>
             <a href="/?timespan=43200">Last 12 hours</a> |
             <a href="/?timespan=10800">Last 3 hours</a> |
             <a href="/?timespan=3600">Last hour</a> <br>
             % for f in figures :
             <img src="data:image/png;base64,{{f}}" alt="Temperature" width=90%><br>
             % end
            </p>
    </div>
  </body>
</html>
